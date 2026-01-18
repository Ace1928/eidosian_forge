from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def back_adjust(data):
    """ back-adjusted data to mimic true historical prices """
    col_order = data.columns
    df = data.copy()
    ratio = df['Adj Close'] / df['Close']
    df['Adj Open'] = df['Open'] * ratio
    df['Adj High'] = df['High'] * ratio
    df['Adj Low'] = df['Low'] * ratio
    df.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1, inplace=True)
    df.rename(columns={'Adj Open': 'Open', 'Adj High': 'High', 'Adj Low': 'Low'}, inplace=True)
    return df[[c for c in col_order if c in df.columns]]