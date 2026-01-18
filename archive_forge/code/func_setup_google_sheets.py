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
def setup_google_sheets(credentials_path: str, ticker: str='NASDAQ:AAPL', start_date: str='2003-01-01', end_date: str='2023-01-01') -> str:
    """
    Creates a new Google Sheet and populates it with historical stock data using the GOOGLEFINANCE function.

    Parameters:
    - credentials_path (str): The path to the JSON file containing the Google Sheets API credentials.
    - ticker (str): The stock ticker symbol to fetch historical data for. Default is "NASDAQ:AAPL".
    - start_date (str): The start date for the historical data in YYYY-MM-DD format. Default is "2003-01-01".
    - end_date (str): The end date for the historical data in YYYY-MM-DD format. Default is "2023-01-01".

    Returns:
    - str: The URL of the newly created Google Sheet containing the historical stock data.

    Example:
    >>> setup_google_sheets("path/to/credentials.json", "NASDAQ:GOOGL", "2020-01-01", "2021-01-01")
    """
    try:
        credentials = credentials.from_authorized_user_file(credentials_path, ['https://www.googleapis.com/auth/spreadsheets'])
        client = gspread.authorize(credentials)
        sheet_title = f'{ticker.replace(':', '_')} Stock Data'
        sh = client.create(sheet_title)
        worksheet = sh.get_worksheet(0)
        finance_formula = f'=GOOGLEFINANCE("{ticker}", "close", DATE({start_date}), DATE({end_date}), "DAILY")'
        worksheet.update('A1', finance_formula)
        return sh.url
    except Exception as e:
        logging.error(f'Failed to setup Google Sheet for {ticker}: {e}')
        raise