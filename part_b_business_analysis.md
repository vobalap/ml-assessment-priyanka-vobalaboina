B1. Problem Formulation — 8 marks
(a) Formulating This as a Machine Learning Problem
Target Variable: 
The target variable is the promotion type to deploy at each store each month — one of 5 categories: Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. The model learns which promotion leads to the maximum number of items sold.
Candidate Input Features:
•	Store size (small, medium, large)
•	Location type (urban, semi-urban, rural)
•	Monthly footfall (number of customers visiting the store)
•	Local competition density (number of nearby competing stores)
•	Customer demographics (age group, income level)
•	Past promotion performance (items sold per promotion type historically)
•	Month or season (to capture seasonal buying patterns)
Type of ML Problem: 
This is a Multi-Class Classification problem.
Justification:
•	The output is one of 5 fixed promotion categories, not a continuous number, so this is classification and not regression.
•	Since there are more than 2 possible output classes, it is specifically multi-class classification.
•	Historical data on past promotions and items sold can be used to train a supervised learning model.
•	The goal is to select the best promotion for each store, not to predict an exact sales figure.
b) Why Items Sold is a More Reliable Target Variable than Total Sales Revenue
Why Revenue is Problematic: 
Total sales revenue is influenced by the price of items, not just the number of items sold. For example, a Flat Discount promotion directly reduces the price of each item, which lowers revenue even if many more items are sold. This means revenue can appear low for a promotion that is actually very effective at driving sales volume. The model would incorrectly learn that discount-based promotions perform poorly, simply because they reduce the price.
Why Items Sold is Better: 
Items sold (sales volume) measures how many products customers actually purchased, regardless of the price paid. This gives a fair and consistent comparison across all five promotion types. Whether a store runs a Flat Discount or a BOGO offer, the number of items sold reflects the true customer response to that promotion without being distorted by pricing effects.
Broader Principle: Target Variable Selection in ML: 
This illustrates the principle that the target variable must directly and cleanly represent the business goal, without being influenced by confounding factors. In real-world ML projects, a poorly chosen target variable can introduce bias into the model and lead to misleading predictions, even when the data and algorithm are correct. The right target variable should be measurable, consistent across all conditions, and free from distortions caused by other variables in the problem.
(c) Alternative Modelling Strategy
The Problem with a Single Global Model
A single global model trained in all 50 stores assumes that every store responds to promotions in the same way. This is not realistic. An urban store with high footfall and strong competition will behave very differently from a small rural store with a loyal but limited customer base. A global model would average out these differences and produce predictions that are too generic to be useful for any specific store.
Proposed Strategy: Location-Based Segmented Models
A better approach is to group stores into segments based on their location type and train a separate model for each segment. For example, one model for urban stores, one for semi-urban stores, and one for rural stores. Each model learns the promotion patterns specific to that group of stores, making predictions more accurate and relevant.
Why This Works Better:
1.Urban stores may respond best to BOGO or Flat Discounts due to high competition and price-sensitive shoppers.
2.Rural stores may respond better to Loyalty Points Bonus due to a smaller but more regular customer base.
3.Each segmented model captures these local patterns without being diluted by data from very different store types.
4. Broader Principle This Illustrates
This approach reflects the principle of stratified or hierarchical modelling — when the data contains distinct subgroups with different underlying behaviors, training separate models per subgroup almost always outperforms a single one-size-fits-all model. It is a common and important strategy in real-world ML deployments.


B2. Data and EDA Strategy — 10 marks
(a) Joining the Tables and Preparing the Modelling Dataset
The Four Tables and How to Join Them
The raw data arrives at four separate tables. Here is how they connect to each other:
•	The transactions table contains one row per purchase and includes a store ID, a date, and the items sold. This is the core table.
•	The store attributes table contains one row per store with details like store size, location type, footfall, and competition density. It joins to transactions on store ID.
•	The promotion details table contains one row per promotion per store per month, describing which promotion was active. It joins transactions on store ID and month.
•	The calendar table contains one row per date with flags for weekends and festivals. It joins to transactions on date.
All four tables are joined together using store ID and date as the common keys, resulting in one unified dataset.
Grain of the Final Modelling Dataset
The grain of the final dataset is one row per store per month. This means each row represents a single store in a single month, along with the promotion that was run and the total items sold during that month.
Aggregations to Perform Before Modelling
Since the transactions table is at the daily or per-purchase level, the following aggregations are needed to bring everything to the monthly store level:
•	Sum up total items sold per store per month from the transactions table.
•	Count the number of weekend days and festival days in that month from the calendar table and add these as numeric features.
•	Carry over store-level attributes like size, location type, footfall, and competition density directly since they do not change per transaction.
•	Attach the promotion type that was active for that store in that month from the promotion details table.
The final dataset will have one row per store per month, with all store attributes, calendar features, the promotion type as the target variable, and total items sold as the outcome used to determine which promotion performed best.
(b) EDA Strategy Before Building the Model
What is EDA and Why It Matters
Exploratory Data Analysis (EDA) is the process of understanding your data before building any model. It helps you spot patterns, catch problems, and make smarter decisions about which features to use and how to engineer them.
Here are four key analyses to perform:
Analysis 1: Distribution of Items Sold per Promotion Type
Plot or calculate the average items sold for each of the five promotion types across all stores. Look for which promotions consistently drive higher sales volume and which ones underperform. If one promotion type has very high variance in sales, it may mean its effectiveness depends heavily on store type, which would justify building segmented models rather than a single global model.
Analysis 2: Promotion Performance by Location Type
Break down average items sold by promotion type and location type together (urban, semi-urban, rural). Look for interaction effects, for example, whether BOGO works well in urban stores but poorly in rural ones. If strong differences exist, location type becomes a critical feature and may even justify training separate models per location segment as discussed in B1(c).
Analysis 3: Correlation Between Store Features and Items Sold
Calculate the correlation between numerical features like footfall, store size, and competition density against items sold. Look for features that have a strong positive or negative relationship with sales volume. Features with high correlation are strong candidates to include in the model, while features with near-zero correlation may be dropped to reduce noise.
Analysis 4: Seasonal Trends Using the Calendar Features
Analyse how items sold vary across months and whether festival flags or weekend counts in a month are associated with higher sales. Look for spikes during festive months or holiday periods. If strong seasonal patterns exist, month of the year, festival count, and weekend count should be included as engineered features in the model to help it account for time-based variation.
Summary of How EDA Influences Modelling Decisions
•	If promotion performance varies by location, use segmented models or add interaction features.
•	If footfall and store size are highly correlated with sales, prioritise them as input features.
•	If seasonal spikes are visible, engineer month and festival flag features explicitly.
•	If any promotion type has very few observations, consider oversampling or flagging class imbalance before training.
(c) Handling Class Imbalance in Promotion Data
Understanding the Problem
In this dataset, 80% of transactions happened without any promotion. This means the model sees very few examples of promoted transactions during training. As a result, the model becomes biased towards predicting no promotion because that is what it sees most of the time. It may appear to perform well overall, but it will actually be very poor at identifying which promotion works best, which is the entire goal of this project.
How Imbalance Affects the Model
•	The model will be trained mostly on non-promotional behaviour and will not learn enough about how each of the five promotions performs.
•	It may default to predicting the majority class (no promotion) most of the time, making it useless for the actual business decision.
•	Standard accuracy will be misleadingly high even if the model never correctly predicts the right promotion type.
Steps to Address the Imbalance
•	Filter the dataset to include only rows where a promotion was active before training the model. Since the goal is to choose the best promotion among the five options, non-promotional transactions are not directly relevant to the classification task.
•	If non-promotional data is kept for context, use oversampling techniques such as SMOTE (Synthetic Minority Oversampling Technique) to generate more synthetic examples for underrepresented promotion types.
•	Use evaluation metrics like F1-score or macro-averaged precision and recall instead of plain accuracy, since accuracy is misleading when classes are imbalanced.
•	Assign class weights during model training so that the model penalises mistakes on minority promotion classes more heavily, forcing it to pay more attention to them.
Key Takeaway
Class imbalance is a very common real-world problem. The most important first step is to recognize it during EDA and then consciously decide how to handle it before modelling, rather than discovering it after the model has already been built and deployed.


B3. Model Evaluation and Deployment — 12 marks
(a) Train-Test Split, Evaluation Metrics, and Interpretation
Why a Random Split is Inappropriate
This dataset has a time dimension — data is collected monthly over three years. A random split would mix past and future data together, meaning the model could accidentally train on data from month 24 and test on data from month 6. This is called data leakage. In real life, you can never use future information to predict the past, so a random split would give falsely optimistic results that would not hold up in deployment.
How to Set Up the Train-Test Split
Use a time-based split that respects the chronological order of the data. A good approach for three years of monthly data is:
•	Training set: Month 1 to Month 30 (first 2.5 years)
•	Validation set: Month 31 to Month 33 (next 3 months, used for tuning the model)
•	Test set: Month 34 to Month 36 (final 3 months, used only for final evaluation)
This ensures the model is always trained on past data and evaluated on future data, which mirrors how it would actually be used in production.
Evaluation Metrics to Use
Since this is a multi-class classification problem, the following metrics are appropriate:
•	Accuracy: The percentage of stores where the model correctly recommends the best promotion. It gives a quick overall sense of performance but can be misleading if some promotion types appear more often than others in the data.
•	Macro-Averaged F1-Score: This calculates the F1-score for each promotion type separately and then averages them equally. It is more reliable than accuracy when promotion classes are imbalanced because it treats each promotion type as equally important regardless of how often it appears in the data.
•	Confusion Matrix: A table showing how often each promotion type was correctly predicted versus confused with another. In a business context, this helps identify which promotions the model struggles to distinguish, for example, if it frequently confuses Flat Discount with BOGO, the marketing team should be aware of this uncertainty.
•	Business Uplift Validation: Beyond standard ML metrics, it is important to check whether stores that followed the model's recommendation actually sold more items than stores that did not. This connects model performance back to the real business goal.
Key Takeaway
A time-based split preserves the integrity of the evaluation by simulating real deployment conditions. Macro F1-score is the most trustworthy metric here because it accounts for class imbalance and treats all five promotion types fairly.
(b) Investigating and Communicating Different Recommendations Using Feature Importance
The Scenario
The model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount for Store 12 in March. Even though it is the same store, the recommendations differ because the input features change across months. Feature importance helps us understand which features drove each decision.
What is Feature Importance
Feature importance is a technique that tells us how much each input feature contributed to the model's prediction. A higher importance score means that feature had a stronger influence on the recommendation made.
How to Investigate the Different Recommendations
After training the model, extract the feature importance scores for Store 12 separately for December and March. Then compare which features were most influential in each month.
For December, the model likely gave high importance to:
•	Festival flag being active (December has Christmas and year-end festivities)
•	High weekend count in the month
•	Historical data showing that loyal customers shop more during festive seasons
These factors together signal that customers in December are already motivated to buy, so rewarding them with Loyalty Points encourages repeat visits and long-term retention rather than just a one-time discount.
For March, the model likely gave high importance to:
•	No festival flags active
•	Lower footfall compared to December
•	Historical data showing that flat discounts drive higher volume during slow months
These factors signal that customers in March need a stronger immediate incentive to make a purchase, making a Flat Discount more effective at driving sales volume.
How to Communicate This to the Marketing Team
Avoid using technical jargon when presenting to a non-technical audience. Instead, frame the explanation in business terms:
•	In December, the model sees high festive activity and loyal customer engagement, so it recommends Loyalty Points Bonus to reward and retain customers who are already shopping.
•	In March, the model sees lower footfall and no festive triggers, so it recommends Flat Discount to attract price-sensitive customers and boost sales volume during a quieter period.
You can support this with a simple table showing the top three features and their important scores for each month side by side, making it easy for the marketing team to see exactly what changed between the two months.
Key Takeaway
Feature importance bridges the gap between a model's mathematical output and a human-understandable business explanation. It builds trust with stakeholders by showing that the model's recommendations are driven by logical, interpretable business signals and not just a black box.
(c) End-to-End Deployment Process
Step 1: Saving the Trained Model
Once the model is trained and evaluated, it is saved as a serialised file using a tool like pickle or joblib in Python. This means the model's learned parameters are stored on disk so it can be loaded and used at any time without retraining. Along with the model file, save the preprocessing pipeline (such as the label encoder for promotion types and the scaler for numerical features) so that new data can be transformed in exactly the same way as the training data.
Step 2: Preparing New Monthly Data
At the start of every month, the following steps are performed to prepare the input data for all 50 stores:
•	Pull the latest store attributes such as footfall, competition density, and demographics from the store database.
•	Pull the calendar features for the upcoming month such as weekend count and festival flags.
•	Pull the promotion history from the previous months to include as lag features if used during training.
•	Apply the same preprocessing pipeline that was saved during training to transform the raw data into the format the model expects.
The result is a clean input table with one row per store, ready to be fed into the model.
Step 3: Generating Monthly Recommendations
Load the saved model file and pass the prepared input table through it. The model outputs a recommended promotion type for each of the 50 stores. These recommendations are then formatted into a simple report or dashboard and shared with the marketing team at the start of each month.
Step 4: Monitoring for Performance Degradation
Even though the model is not retrained every month, its performance must be monitored regularly. The following monitoring steps should be put in place:
•	Track prediction accuracy each month by comparing the model's recommended promotion against the promotion that actually performed best in hindsight, once the month's sales data is available.
•	Monitor data drift by checking whether the distribution of input features such as footfall or competition density has shifted significantly compared to the training data. If the input data starts looking very different from what the model was trained on, predictions will become unreliable.
•	Set a performance threshold, for example, if macro F1-score drops below 0.65 for two consecutive months, trigger a retraining process.
•	Monitor business outcomes directly by tracking whether stores following the model's recommendations are consistently achieving higher items sold compared to stores that did not follow the recommendation.
Step 5: When to Retrain
Retrain the model when any of the following occur:
•	Model accuracy drops below the defined threshold for two or more consecutive months.
•	A significant data drift is detected in key input features.
•	The business introduces a new promotion type or changes store operations significantly.
•	A full year of new data has accumulated since the last training cycle.
