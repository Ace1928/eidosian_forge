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
def optimize_strategy(data: pd.DataFrame, strategy_func: Callable, parameter_ranges: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
    """

    Optimizes the parameters of a trading strategy using grid search.

    Parameters:

    - data (pd.DataFrame): The historical price data for optimization.

    - strategy_func (Callable): The trading strategy function to be optimized.

    - parameter_ranges (Dict[str, Tuple[Any, Any]]): The ranges of parameter values to search.

    Returns:

    - Dict[str, Any]: The optimal parameters for the trading strategy.

    """
    best_parameters = {}
    best_roi = -np.inf
    parameter_combinations = list(product(*parameter_ranges.values()))
    for combination in parameter_combinations:
        parameters = {key: value for key, value in zip(parameter_ranges.keys(), combination)}
        roi = backtest_strategy(data, strategy_func, parameters)
        if roi > best_roi:
            best_roi = roi
            best_parameters = parameters
    return best_parameters