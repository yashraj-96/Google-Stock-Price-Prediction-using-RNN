# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:19:08 2020

@author: Yashraj
"""
#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
Dtrain = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = Dtrain.iloc[:,1:2].values

#Feature Scaling (Normalization technique for RNN)
from sklearn.preprocessing import MinMaxScaler
nm = MinMaxScaler(feature_range=(0,1))
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = nm.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60,1258):
   x_train.append(training_set_scaled[i-60:i,0])
   y_train.append(training_set_scaled[i,0])
x_train , y_train = np.array(x_train), np.array(y_train)


#Reshaping 
x_train = np.reshape(x_train, (x_train.shape[0],
                             x_train.shape[1],
                             1))

#Building the Recurrent Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding first LSTM layer along with dropout
regressor.add(LSTM(units=50, return_sequences=True,
                   input_shape= (x_train.shape[1], 1))x`)
regressor.add(Dropout(p=0.2))

#Adding the second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(p=0.2))

#Adding the third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(p=0.2))

#Adding the fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(p=0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the layers
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN
regressor.fit(x_train, y_train , epochs = 100 , batch_size=32)
    
#Predicting the results

#Getting the real stock price of 2017
Dtest = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = Dtest.iloc[:,1:2].values

#Getting the predicted stock price
total_data = pd.concat((Dtrain['Open'], Dtest['Open']), axis = 0)
inputs = total_data[len(total_data)-len(Dtest) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = nm.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
predicted_stock_price=regressor.predict(x_test)
predicted_stock_price = nm.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price , color= 'red' , label='real_stock_price')
plt.plot(predicted_stock_price , color= 'green', label='predicted_stock_price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show











