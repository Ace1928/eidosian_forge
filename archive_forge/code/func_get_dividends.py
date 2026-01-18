import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
def get_dividends(self, proxy=None) -> pd.Series:
    if self._history is None:
        self.history(period='max', proxy=proxy)
    if self._history is not None and 'Dividends' in self._history:
        dividends = self._history['Dividends']
        return dividends[dividends != 0]
    return pd.Series()