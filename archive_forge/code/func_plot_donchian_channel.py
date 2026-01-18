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
def plot_donchian_channel(data: pd.DataFrame, window: int=300) -> None:
    """
    Plots the Donchian Channel for a given DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the price data and Donchian Channel.
    - window (int): The window size to plot the Donchian Channel for.
    """
    if 'ticker' in data.columns:
        sample_ticker = data['ticker'].iloc[0]
    else:
        sample_ticker = 'Unknown Ticker'
    plt.plot(data[-window:].close, label='CLOSE', color='blue')
    plt.plot(data[-window:].dcl, color='black', linestyle='--', alpha=0.3, label='DCL')
    plt.plot(data[-window:].dcm, color='orange', label='DCM')
    plt.plot(data[-window:].dcu, color='black', linestyle='--', alpha=0.3, label='DCU')
    plt.legend()
    plt.title(f'{sample_ticker} Donchian Channels Over Last {window} Days', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.show(block=False)
    plt.pause(10)
    plt.savefig(f'{sample_ticker}_donchian_channel_{time.time()}.png')
    try:
        plt.close()
    except Exception as e:
        print(f'An error occurred while closing the plot: {str(e)}')