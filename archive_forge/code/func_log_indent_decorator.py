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
def log_indent_decorator(func):

    def wrapper(*args, **kwargs):
        logger = get_indented_logger('yfinance')
        logger.debug(f'Entering {func.__name__}()')
        with IndentationContext():
            result = func(*args, **kwargs)
        logger.debug(f'Exiting {func.__name__}()')
        return result
    return wrapper