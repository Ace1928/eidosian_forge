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
def ten_day_average_volume(self):
    if self._10d_avg_vol is not None:
        return self._10d_avg_vol
    prices = self._get_1y_prices(fullDaysOnly=True)
    if prices.empty:
        self._10d_avg_vol = None
    else:
        n = prices.shape[0]
        a = n - 10
        b = n
        if a < 0:
            a = 0
        self._10d_avg_vol = int(prices['Volume'].iloc[a:b].mean())
    return self._10d_avg_vol