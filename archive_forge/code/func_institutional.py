import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
@property
def institutional(self) -> pd.DataFrame:
    if self._institutional is None:
        self._fetch_and_parse()
    return self._institutional