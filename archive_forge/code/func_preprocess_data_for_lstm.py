import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to a specified URL
import pandas_ta as ta  # For technical analysis indicators
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from termcolor import colored as cl  # For coloring terminal text
import math  # Provides access to mathematical functions
import logging  # For tracking events that happen when some software runs
import nltk  # For natural language processing tasks
from newspaper import Article  # For extracting and parsing news articles
import ccxt  # Cryptocurrency exchange library for connecting to various exchanges
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import gspread
from oauth2client.client import OAuth2Credentials
from tkinter import Tk, Label, Entry, Button, StringVar
import json
import os
import yfinance as yf
from itertools import product
from concurrent.futures import ThreadPoolExecutor, wait
import time
import yaml
import traceback
from typing import TypeAlias
@log_exception
@log_function_call
def preprocess_data_for_lstm(data: pd.DataFrame, feature_columns: List[str], target_column: str, time_steps: int=60) -> Tuple[np.ndarray, np.ndarray]:
    """

    Prepares data for LSTM model training and prediction by scaling the features and creating sequences of time steps.

    Parameters:

    - data (pd.DataFrame): The DataFrame containing the data.

    - feature_columns (List[str]): List of column names to be used as features.

    - target_column (str): Name of the target column.

    - time_steps (int): The number of time steps to be used for training (default is 60).

    Returns:

    - Tuple[np.ndarray, np.ndarray]: Tuple containing feature and target data arrays.

    """
    data = data[feature_columns + [target_column]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = ([], [])
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, :-1])
        y.append(scaled_data[i, -1])
    return (np.array(X), np.array(y))