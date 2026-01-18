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
def format_quarterly_financial_statement(_statement, level_detail, order):
    """
    format_quarterly_financial_statements formats any quarterly financial statement

    Returns:
        - _statement: A fully formatted quarterly financial statement in pandas dataframe.
    """
    _statement = _statement.reindex(order)
    _statement.index = camel2title(_statement.T)
    _statement['level_detail'] = level_detail
    _statement = _statement.set_index([_statement.index, 'level_detail'])
    _statement = _statement[sorted(_statement.columns, reverse=True)]
    _statement = _statement.dropna(how='all')
    _statement.columns = _pd.to_datetime(_statement.columns).date
    return _statement