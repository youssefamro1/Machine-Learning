                
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
regr=LinearRegression()
regr.fit(X_train, y_train)

y_pred=regr.predict(X_test)
y_pred_train=regr.predict(X_train)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,y_pred_train,color="blue")
plt.title('Salary vs Experience (train set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,y_pred_train,color="blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()









