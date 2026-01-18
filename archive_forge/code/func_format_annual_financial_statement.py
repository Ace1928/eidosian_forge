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
def format_annual_financial_statement(level_detail, annual_dicts, annual_order, ttm_dicts=None, ttm_order=None):
    """
    format_annual_financial_statement formats any annual financial statement

    Returns:
        - _statement: A fully formatted annual financial statement in pandas dataframe.
    """
    Annual = _pd.DataFrame.from_dict(annual_dicts).set_index('index')
    Annual = Annual.reindex(annual_order)
    Annual.index = Annual.index.str.replace('annual', '')
    if ttm_dicts and ttm_order:
        TTM = _pd.DataFrame.from_dict(ttm_dicts).set_index('index').reindex(ttm_order)
        TTM.columns = ['TTM ' + str(col) for col in TTM.columns]
        TTM.index = TTM.index.str.replace('trailing', '')
        _statement = Annual.merge(TTM, left_index=True, right_index=True)
    else:
        _statement = Annual
    _statement.index = camel2title(_statement.T.index)
    _statement['level_detail'] = level_detail
    _statement = _statement.set_index([_statement.index, 'level_detail'])
    _statement = _statement[sorted(_statement.columns, reverse=True)]
    _statement = _statement.dropna(how='all')
    return _statement