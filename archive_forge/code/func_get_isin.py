from __future__ import print_function
from io import StringIO
import json as _json
import warnings
from typing import Optional, Union
from urllib.parse import quote as urlencode
import pandas as pd
import requests
from . import utils, cache
from .data import YfData
from .scrapers.analysis import Analysis
from .scrapers.fundamentals import Fundamentals
from .scrapers.holders import Holders
from .scrapers.quote import Quote, FastInfo
from .scrapers.history import PriceHistory
from .const import _BASE_URL_, _ROOT_URL_
def get_isin(self, proxy=None) -> Optional[str]:
    if self._isin is not None:
        return self._isin
    ticker = self.ticker.upper()
    if '-' in ticker or '^' in ticker:
        self._isin = '-'
        return self._isin
    q = ticker
    self._quote.proxy = proxy or self.proxy
    if self._quote.info is None:
        return None
    if 'shortName' in self._quote.info:
        q = self._quote.info['shortName']
    url = f'https://markets.businessinsider.com/ajax/SearchController_Suggest?max_results=25&query={urlencode(q)}'
    data = self._data.cache_get(url=url, proxy=proxy).text
    search_str = f'"{ticker}|'
    if search_str not in data:
        if q.lower() in data.lower():
            search_str = '"|'
            if search_str not in data:
                self._isin = '-'
                return self._isin
        else:
            self._isin = '-'
            return self._isin
    self._isin = data.split(search_str)[1].split('"')[0].split('|')[0]
    return self._isin