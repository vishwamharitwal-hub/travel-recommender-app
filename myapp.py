import pandas as pd 
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the datasets

# demond file 
df_demond = pd.read_csv("data/travel_demand.csv", encoding='utf-8-sig')
# remomende file 
df = pd.read_csv("data/india_travel.csv", encoding='unicode_escape')


# clean code 

df['Type']=df['Type'].str.lower().str.strip()
df['Country']=df['Country'].str.lower().str.strip()
df.columns = df.columns.str.replace("ï»¿", "").str.strip()  
df_demond['Month']=pd.to_datetime(df_demond['Month'],format='%b-%y')
df_demond['Month_num']=df_demond['Month'].dt.month
df_demond['Destination']=df_demond['Destination'].str.lower().str.lower()





#  Ml model training
le = LabelEncoder()
df_demond['Destination_encoded'] = le.fit_transform(df_demond['Destination'])



# Ml model traning 
x=df_demond[['Month_num','Destination_encoded']]
y=df_demond['Demand (visits, reviews, searches)']

model= LinearRegression()
model.fit(x,y)

st.write("Encoder trained on destinations:", list(le.classes_))



# Recommendation Function
def recomended_place(country=None,Type=None,Best_Visit_Months=None):
    new_df=df.copy()
    if country:
       new_df=new_df[new_df['Country'].str.contains(country.strip().lower(),na=False)]
    if Type:
       new_df=new_df[new_df['Type'].str.contains(Type.strip().lower(),na=False)]
    if Best_Visit_Months:
        new_df=new_df[new_df['BestTimeToVisit'].str.contains(Best_Visit_Months.strip().lower(),case=False)]
    return new_df[['Destination' ,'Country', 'Type',  'BestTimeToVisit']]





# Streamlit app
st.set_page_config(page_title="Travel Destination Recommender",page_icon="🌍", layout="wide")
st.title("🌍Travel Destination Recommender & Demond app ")
st.markdown("use this app to ** recomend destinations**and**predict travel demand** based on your preferences.")
st.subheader(" 📌1.Travel Demand Prediction")

col1, col2,col3 = st.columns(3)

with col1:
    country=st.text_input("Enter a country (optional):")
with col2:
    Type=st.text_input("Enter a type of destination (e.g., beach, mountain, city)(optional):")
with col3:
    Best_Visit_Months=st.text_input("Enter best visit months (optional):")

if st.button("Get Recommendations"):
    result = recomended_place(country, Type, Best_Visit_Months)
    if not result.empty:
        st.success("✅ Recommended Travel Destinations:")
        st.dataframe(result)
    else:
        st.warning("No recommendations found based on your criteria.")

# demond prediction 
st.subheader(" 📌2.Travel Demand Prediction")

col4, col5 = st.columns(2)

with col4:
    month= st.number_input("Enter month (1-12):", min_value=1, max_value=12, value=5)

with col5:
    destination = st.text_input("Enter destination (e.g., Goa):", value='Goa')
    dest_input = destination.strip().lower()

#  prediction button 

if st.button("Predict Demand"):
    dest_input = destination.strip().lower()
    if dest_input in le.classes_:
        destination_encoded = le.transform([dest_input])[0]
        prediction = model.predict([[ int(month), destination_encoded]])
        st.success(f"📊 Predicted demand for **{destination.title()}** in month **{month}** is **{int(prediction[0])}**")
    else:   
        st.error(f"❌ '{destination}' not found in dataset.\nTry one of: {list(le.classes_)}")

        
    

# Plotting the demand data
if st.button("Show Demand Plot"):
    matched_dest=next((d for d in le.classes_ if isinstance(d, str) and d.lower() == dest_input), None)

    if matched_dest:
        month__list=list(range(1, 13))
        destination_encoded= le.transform([matched_dest])[0]
        demand_values = [model.predict([[m, destination_encoded]])[0] for m in month__list]

# ploting the demand data
        fig, ax =plt.subplots() 
        ax.bar(month__list, demand_values,color='skyblue')
        ax.set_title(f"Demand for {dest_input.title()} by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Demand (visits, reviews, searches)")
        st.pyplot(fig)
    else:
        st.error(f"❌ '{destination}' not found in dataset.\nTry one of: {list(le.classes_)}")
        



