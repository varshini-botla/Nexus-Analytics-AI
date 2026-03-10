import streamlit as st
import pandas as pd
import json
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from etl import extract, transform, load_to_dw, export_outputs
import time
import os

# ---- CONFIG & SETTINGS ----
st.set_page_config(
    page_title="Nexus Analytics AI",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---- UTILS ----
def get_db_path():
    return BASE / "warehouse.db"

def save_uploaded_file(uploaded_file):
    try:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext == ".csv":
            target_path = DATA_DIR / "Sample - Superstore.csv"
            return target_path, "Structured (CSV)", "📄"
        elif file_ext == ".json":
            target_path = DATA_DIR / "orders.json"
            return target_path, "Semi-Structured (JSON)", "📜"
        elif file_ext == ".txt":
            target_path = DATA_DIR / "reviews.txt"
            return target_path, "Unstructured (TXT)", "📝"
        else:
            return None, "Unknown", "❓"
    except Exception:
        return None, "Error", "❌"

# ---- PREVENT RE-RUN ETL ON EVERY OVERLAY ----
if 'etl_done' not in st.session_state:
    st.session_state.etl_done = False

# ---- NEXUS GLASSMORPHISM CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --secondary: #a855f7;
    --accent: #22d3ee;
    --bg: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.7);
    --border: rgba(255, 255, 255, 0.1);
}

.stApp {
    background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}

/* Glass Card Effect */
.nexus-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.nexus-card:hover {
    border: 1px solid rgba(99, 102, 241, 0.4);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    transform: translateY(-2px);
}

/* Typography Gradient */
.gradient-h1 {
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3.5rem;
    letter-spacing: -0.05em;
    margin-bottom: 0.5rem;
}

.nexus-accent {
    color: var(--accent);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.8rem;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid var(--border);
}

/* Custom Metric Design */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 1rem;
}

[data-testid="stMetricValue"] {
    color: #fff !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    opacity: 0.9;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}

</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 2rem;">
            <div style="background: linear-gradient(135deg, #6366f1, #a855f7); width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white;">N</div>
            <div style="line-height: 1.1;">
                <div style="font-weight: 700; color: white; font-size: 1.2rem;">NEXUS</div>
                <div style="font-size: 0.7rem; color: #94a3b8; letter-spacing: 0.1em;">ANALYTICS AI</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["Intelligence Hub", "Predictive Suite", "Data Factory", "Core Warehouse"],
        label_visibility="hidden"
    )
    
    st.markdown("---")
    st.markdown('<div class="nexus-accent">System Node</div>', unsafe_allow_html=True)
    status = "Online" if get_db_path().exists() else "Standby"
    st.caption(f"Status: {status}")
    st.caption("Version: 2.0.0 (Nexus)")

# ---- PAGE ROUTING ----
try:
    conn = sqlite3.connect(get_db_path()) if get_db_path().exists() else None
    if page == "Intelligence Hub":
        st.markdown('<div class="nexus-accent">Overview</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-h1">Intelligence Hub</h1>', unsafe_allow_html=True)
        
        if not conn:
            st.warning("Nexus Node Standby: Please initialize data in Data Factory.")
        else:
            # KPI Extraction
            sales_val = pd.read_sql("SELECT SUM(sales) as v FROM fact_sales", conn).iloc[0]['v'] or 0
            orders_val = pd.read_sql("SELECT COUNT(DISTINCT order_id) as v FROM fact_sales", conn).iloc[0]['v'] or 0
            cust_val = pd.read_sql("SELECT COUNT(DISTINCT customer_id) as v FROM dim_customer", conn).iloc[0]['v'] or 0
            
            # Sentiment Analysis (Advanced)
            sent_df = pd.read_sql("SELECT sentiment FROM dim_review", conn)
            score = int((sent_df[sent_df['sentiment']=='positive'].shape[0] / len(sent_df) * 100)) if not sent_df.empty else 0

            # HERO METRICS
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Revenue", f"${sales_val:,.0f}", "+12.5%")
            with m2:
                st.metric("Total Orders", f"{orders_val:,}", "+5.2%")
            with m3:
                st.metric("Unique Customers", f"{cust_val:,}", "+8.1%")
            with m4:
                st.metric("Sentiment Score", f"{score}%", "+2.4%")

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
                st.subheader("Market Distribution")
                cat_df = pd.read_sql("SELECT dp.category, SUM(fs.sales) as s FROM fact_sales fs JOIN dim_product dp ON fs.product_id=dp.product_id GROUP BY dp.category", conn)
                fig = px.bar(cat_df, x='category', y='s', color='category',
                             color_discrete_sequence=px.colors.sequential.Plasma_r,
                             template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
                st.subheader("Customer Segments")
                seg_df = pd.read_sql("SELECT segment, COUNT(*) as counts FROM ml_customer_segments GROUP BY segment", conn)
                if not seg_df.empty:
                    fig2 = px.pie(seg_df, values='counts', names='segment', hole=0.6,
                                 color_discrete_sequence=['#6366f1', '#a855f7', '#22d3ee'])
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=True)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.caption("No segments generated yet.")
                st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Predictive Suite":
        st.markdown('<div class="nexus-accent">AI Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-h1">Predictive Suite</h1>', unsafe_allow_html=True)
        
        if not conn:
            st.info("Upload data in Data Factory to see AI forecasts.")
        else:
            st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
            st.subheader("30-Day Sales Projection")
            forecast_df = pd.read_sql("SELECT * FROM ml_sales_forecast", conn)
            if not forecast_df.empty:
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_sales'], 
                                         mode='lines+markers', name='Predicted',
                                         line=dict(color='#22d3ee', width=3)))
                fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Date", yaxis_title="Predicted Sales ($)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Insufficient historical data for forecasting. Please process more sales records.")
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Data Factory":
        st.markdown('<div class="nexus-accent">Processing Unit</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-h1">Data Factory</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.subheader("Nexus Ingestion")
        uploaded_file = st.file_uploader("Inject Raw Data (Nexus Multi-Format Support)", type=["csv", "json", "txt"])
        
        if uploaded_file:
            path, dtype, icon = save_uploaded_file(uploaded_file)
            if path:
                st.success(f"{icon} System ready to process {dtype}")
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Initialize Neural ETL Pipeline", type="primary"):
                    with st.spinner("Executing Nexus ETL Protocol..."):
                        try:
                            df_raw, o_raw, r_raw = extract(target_type=dtype)
                            sales, prod, cust, o_df, r_df, agg, segments, forecast = transform(df_raw, o_raw, r_raw)
                            load_to_dw(sales, prod, cust, o_df, r_df, agg, segments, forecast)
                            st.session_state.etl_done = True
                            st.balloons()
                            st.success("Nexus Integration Successful")
                        except Exception as e:
                            st.error(f"ETL Failure: {e}")
        else:
            st.info("Drop a CSV, JSON, or TXT file to begin processing.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Core Warehouse":
        st.markdown('<div class="nexus-accent">Raw Buffer</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-h1">Core Warehouse</h1>', unsafe_allow_html=True)
        
        if conn:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
            if tables:
                t = st.selectbox("Select Memory Block", tables)
                df = pd.read_sql(f"SELECT * FROM {t} LIMIT 1000", conn)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Warehouse is online but empty.")
        else:
            st.error("Nexus Storage Offline. Please initialize data.")

except Exception as e:
    st.error(f"Nexus OS Error: {e}")
finally:
    if conn: conn.close()
