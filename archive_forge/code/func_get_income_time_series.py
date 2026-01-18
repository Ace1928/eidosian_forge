import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
def get_income_time_series(self, freq='yearly', proxy=None) -> pd.DataFrame:
    res = self._income_time_series
    if freq not in res:
        res[freq] = self._fetch_time_series('income', freq, proxy)
    return res[freq]