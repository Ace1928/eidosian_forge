import datetime
import json
import warnings
from collections.abc import MutableMapping
import numpy as _np
import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import quote_summary_valid_modules, _BASE_URL_
from yfinance.exceptions import YFNotImplementedError, YFinanceDataException, YFinanceException
@property
def day_high(self):
    if self._day_high is not None:
        return self._day_high
    prices = self._get_1y_prices()
    if prices.empty:
        self._day_high = None
    else:
        self._day_high = float(prices['High'].iloc[-1])
        if _np.isnan(self._day_high):
            self._day_high = None
    return self._day_high