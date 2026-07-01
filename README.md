UCCD2063 – AI Techniques

This project is from the AI Techniques course in my Computer Science degree. The task was to build a complete machine learning pipeline that predicts an employee's salary category from their job and demographic information — train a few different models, compare them, then tune the best one and explain the choice.

I worked on this in a team of three.

What it does

Given features about a tech employee (things like years of experience, projects completed, education level, job title, location, work mode, and more), the model predicts which salary bracket they fall into — High, Medium, or Low. It's a multi-class classification problem, and the focus of the project was doing the whole ML process properly, from raw messy data all the way to a tuned and evaluated final model.

The data

The dataset (Tech_employee_salary.csv) mixes numerical features (experience, projects completed, overtime hours, etc.) and categorical ones (gender, education level, job title, location, work mode, primary language, and others). I started by exploring it — distributions, value counts, and histograms — to understand the features and spot issues before modelling.

How I approached it

I built the pipeline end to end:


Data exploration — inspected distributions, checked categorical value counts, and visualised the features to understand the data.
Train/test split — held out 20% for final testing.
Preprocessing — split features into numerical and categorical, then:

filled missing numerical values using median imputation,
scaled the numerical features with StandardScaler,
one-hot encoded the categorical features,
label-encoded the target.



Trained and compared three models, then tuned the strongest one and evaluated it on the untouched test set.


Models I compared


SGD Classifier
Random Forest Classifier
Logistic Regression


For each, I measured accuracy, ran cross-validation, and looked at the confusion matrix along with macro precision, recall, and F1 score so I wasn't just relying on raw accuracy.

Choosing and tuning the final model

After comparing all three, I chose Logistic Regression as the final model and tuned it with GridSearchCV (5-fold cross-validation), searching over the regularisation strength C and the solver. I then evaluated the tuned model on the held-out test set using accuracy and a confusion matrix.


Results: (fill in from your run) — best hyperparameters, test accuracy, and macro F1. You can copy these straight from the final cells of the notebook.



Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib, and Seaborn.

Running it


Install the dependencies:


bash   pip install scikit-learn pandas numpy matplotlib seaborn


Make sure Tech_employee_salary.csv is in the same folder as the notebook.
Open the notebook and run the cells in order:


bash   jupyter notebook G56.ipynb

What I took away from it

This was the project where the full ML workflow really clicked for me. The model itself is only a small part — most of the real work is in preparing the data well (handling missing values, scaling, encoding) and evaluating honestly with cross-validation and the right metrics rather than just accuracy. Getting the preprocessing and evaluation right mattered more than which algorithm I picked.# UCCD2063-AI-TECHNIQUES
This is a project of a course I have taken in my Bachelor's degree where it needs me to train three different models and tune them and choose one model to continue with an explanation
