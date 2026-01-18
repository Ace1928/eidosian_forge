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
def backtest_strategy(data: pd.DataFrame, strategy_func: Callable, parameters: Dict[str, Any]) -> float:
    """

    Backtests a trading strategy on historical data with the specified parameters.

    Parameters:

    - data (pd.DataFrame): The historical price data for backtesting.

    - strategy_func (Callable): The trading strategy function to be backtested.

    - parameters (Dict[str, Any]): The parameters for the trading strategy.

    Returns:

    - float: The return on investment (ROI) of the trading strategy.

    """
    initial_investment = 10000
    equity = initial_investment
    for i in range(len(data)):
        if i < parameters['window']:
            continue
        equity, = strategyfunc(data[:i], equity, **parameters)
    roi = (equity - initial_investment) / initial_investment * 100
    return roi