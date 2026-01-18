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
def get_stored_data(self, query: str, params: tuple=()) -> List[Dict[str, Any]]:
    """Fetch data from the database using a SELECT query.

        Args:
            query (str): SQL query string for fetching data.
            params (tuple): Parameters to substitute into the query.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a row of query results.
        """
    self.cursor.execute(query, params)
    columns = [column[0] for column in self.cursor.description]
    return [dict(zip(columns, row)) for row in self.cursor.fetchall()]