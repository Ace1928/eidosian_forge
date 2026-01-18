import pandas as pd  # Data manipulation
import requests  # HTTP requests
import pandas_ta as ta  # Technical analysis
import matplotlib as mpl  # Plotting
import matplotlib.pyplot as plt  # Plotting
from termcolor import colored as cl  # Text customization
import math  # Mathematical operations
import numpy as np  # Numerical operations
from datetime import datetime as dt  # Date and time operations
from typing import (
import sqlite3  # Database operations
import yfinance as yf  # Yahoo Finance API
from sqlite3 import Connection, Cursor
from typing import Optional  # Type hinting
import seaborn as sns  # Data visualization
import logging  # Logging
import time  # Time operations
import sys  # System-specific parameters and functions
from scripts.trading_bot.indecache import async_cache  # Async cache decorator
def calculate_donchian_channel(data: DataFrame, lower_length: int=40, upper_length: int=50) -> DataFrame:
    """
    Calculates the Donchian Channel for the given DataFrame.

    Parameters:
    - data (DataFrame): The DataFrame containing the stock price data.
    - lower_length (int): The length of the lower Donchian Channel (default is 40).
    - upper_length (int): The length of the upper Donchian Channel (default is 50).

    Returns:
    - DataFrame: The input DataFrame with the added Donchian Channel columns ('dcl', 'dcm', 'dcu').
    """
    data = pd.DataFrame(data)
    data['dcl'] = data['low'].rolling(window=lower_length).min()
    data['dcu'] = data['high'].rolling(window=upper_length).max()
    data['dcm'] = (data['dcl'] + data['dcu']) / 2
    data = data.dropna().drop('time', axis=1, errors='ignore').rename(columns={'dateTime': 'date'}).set_index('date')
    data.index = pd.to_datetime(data.index)
    return data