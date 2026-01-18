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
def parse_actions(data):
    dividends = None
    capital_gains = None
    splits = None
    if 'events' in data:
        if 'dividends' in data['events']:
            dividends = _pd.DataFrame(data=list(data['events']['dividends'].values()))
            dividends.set_index('date', inplace=True)
            dividends.index = _pd.to_datetime(dividends.index, unit='s')
            dividends.sort_index(inplace=True)
            dividends.columns = ['Dividends']
        if 'capitalGains' in data['events']:
            capital_gains = _pd.DataFrame(data=list(data['events']['capitalGains'].values()))
            capital_gains.set_index('date', inplace=True)
            capital_gains.index = _pd.to_datetime(capital_gains.index, unit='s')
            capital_gains.sort_index(inplace=True)
            capital_gains.columns = ['Capital Gains']
        if 'splits' in data['events']:
            splits = _pd.DataFrame(data=list(data['events']['splits'].values()))
            splits.set_index('date', inplace=True)
            splits.index = _pd.to_datetime(splits.index, unit='s')
            splits.sort_index(inplace=True)
            splits['Stock Splits'] = splits['numerator'] / splits['denominator']
            splits = splits[['Stock Splits']]
    if dividends is None:
        dividends = _pd.DataFrame(columns=['Dividends'], index=_pd.DatetimeIndex([]))
    if capital_gains is None:
        capital_gains = _pd.DataFrame(columns=['Capital Gains'], index=_pd.DatetimeIndex([]))
    if splits is None:
        splits = _pd.DataFrame(columns=['Stock Splits'], index=_pd.DatetimeIndex([]))
    return (dividends, splits, capital_gains)