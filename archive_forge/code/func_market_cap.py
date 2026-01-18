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
def market_cap(self):
    if self._mcap is not None:
        return self._mcap
    try:
        shares = self.shares
    except Exception as e:
        if 'Cannot retrieve share count' in str(e):
            shares = None
        elif 'failed to decrypt Yahoo' in str(e):
            shares = None
        else:
            raise
    if shares is None:
        self._tkr.info
        k = 'marketCap'
        if self._tkr._quote._retired_info is not None and k in self._tkr._quote._retired_info:
            self._mcap = self._tkr._quote._retired_info[k]
    else:
        self._mcap = float(shares * self.last_price)
    return self._mcap