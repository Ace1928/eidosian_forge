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
def recommendations(self) -> pd.DataFrame:
    if self._recommendations is None:
        result = self._fetch(self.proxy, modules=['recommendationTrend'])
        if result is None:
            self._recommendations = pd.DataFrame()
        else:
            try:
                data = result['quoteSummary']['result'][0]['recommendationTrend']['trend']
            except (KeyError, IndexError):
                raise YFinanceDataException(f'Failed to parse json response from Yahoo Finance: {result}')
            self._recommendations = pd.DataFrame(data)
    return self._recommendations