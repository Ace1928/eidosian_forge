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
def last_volume(self):
    if self._last_volume is not None:
        return self._last_volume
    prices = self._get_1y_prices()
    self._last_volume = None if prices.empty else int(prices['Volume'].iloc[-1])
    return self._last_volume