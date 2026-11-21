# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.

The goal is to build a classification model that can distinguish between legitimate and fraudulent transactions based on historical transaction data.

The project includes data preprocessing, exploratory data analysis (EDA), model training, performance evaluation, threshold selection, and model saving.

Two machine learning algorithms were explored:

- Logistic Regression
- Random Forest Classifier

The final Random Forest model achieved an F1-Score of **0.85** for the fraud class on the independent test dataset.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original dataset contains:

- 284,807 transactions
- 30 input features
- 1 target variable (`Class`)

The target variable represents:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of all transactions.

## Dataset Download

The dataset is not included in this repository due to its file size.

You can download the original dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place `creditcard.csv` inside the `data/` directory.

## Project Workflow

The project follows these main steps:

1. **Data Collection:** Loading the credit card transaction dataset.
2. **Data Cleaning:** Checking missing values and removing duplicate records.
3. **Exploratory Data Analysis:** Analyzing transaction distributions, outliers, and feature correlations.
4. **Data Preprocessing:** Separating features and target, splitting the data, and applying feature scaling.
5. **Model Development:** Training Logistic Regression and Random Forest classifiers.
6. **Class Imbalance Analysis:** Exploring SMOTE as an oversampling technique on training data.
7. **Threshold Selection:** Selecting a classification threshold using validation data.
8. **Model Evaluation:** Evaluating the final model on an independent test dataset.
9. **Model Saving:** Saving the trained model, scaler, and model configuration.

## Technologies and Libraries

The following tools and libraries were used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Jupyter Notebook

## Model Development

Two classification algorithms were implemented:

### Logistic Regression

A supervised classification algorithm used as a baseline model for distinguishing legitimate and fraudulent transactions.

### Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

The Random Forest model was selected for the final evaluation.

## Data Splitting Strategy

The cleaned dataset was divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

Stratified splitting was used to preserve the class distribution across the three subsets.

StandardScaler was fitted only on the training data and then applied to the validation and test datasets.

## Final Model Results

The final Random Forest model was evaluated using a classification threshold of 0.5, selected based on validation performance.

### Fraud Detection Performance

| Metric | Score |
|---|---:|
| Precision | 0.93 |
| Recall | 0.77 |
| F1-Score | 0.85 |
| Accuracy | Approximately 99.95% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 42,484 | 4 |
| Actual Fraud | 16 | 55 |

The model correctly identified 55 out of 71 fraudulent transactions in the test dataset.

Precision, Recall, and F1-Score for the fraud class are particularly important because of the highly imbalanced nature of the dataset.

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── data/
│   └── creditcard.csv
│
├── images/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── model_settings.json
│
├── notebooks/
│   └── Data_Understanding.ipynb
│
└── README.md
```

*Note: The filenames above are examples. Adjust them to match the actual files in your project.*

## Model Files

The trained model and preprocessing objects are saved in the `models/` directory.

- `random_forest.pkl`: Trained Random Forest classifier.
- `scaler.pkl`: Fitted StandardScaler.
- `model_settings.json`: Model configuration, including the selected classification threshold.

These files can be loaded for future predictions without retraining the model.

## Limitations

- The dataset is highly imbalanced.
- Most input features are anonymized PCA-transformed variables.
- The dataset contains historical transactions from 2013.
- The model does not identify every fraudulent transaction.
- Further testing and optimization are necessary before real-world deployment.

## Future Improvements

Possible future improvements include:

- Testing additional machine learning algorithms.
- Performing more systematic hyperparameter tuning.
- Exploring cost-sensitive learning techniques.
- Improving fraud detection recall.
- Evaluating the model on newer transaction datasets.
- Developing an interactive fraud detection application.

## Conclusion

This project demonstrates the application of machine learning techniques
