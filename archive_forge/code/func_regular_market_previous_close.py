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
def regular_market_previous_close(self):
    if self._reg_prev_close is not None:
        return self._reg_prev_close
    prices = self._get_1y_prices()
    if prices.shape[0] == 1:
        prices = self._get_1wk_1h_reg_prices()
        prices = prices[['Close']].groupby(prices.index.date).last()
    if prices.shape[0] < 2:
        self._tkr.info
        k = 'regularMarketPreviousClose'
        if self._tkr._quote._retired_info is not None and k in self._tkr._quote._retired_info:
            self._reg_prev_close = self._tkr._quote._retired_info[k]
    else:
        self._reg_prev_close = float(prices['Close'].iloc[-2])
    return self._reg_prev_close