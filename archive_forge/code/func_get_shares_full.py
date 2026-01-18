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
@utils.log_indent_decorator
def get_shares_full(self, start=None, end=None, proxy=None):
    logger = utils.get_yf_logger()
    tz = self._get_ticker_tz(proxy=proxy, timeout=10)
    dt_now = pd.Timestamp.utcnow().tz_convert(tz)
    if start is not None:
        start_ts = utils._parse_user_dt(start, tz)
        start = pd.Timestamp.fromtimestamp(start_ts).tz_localize('UTC').tz_convert(tz)
    if end is not None:
        end_ts = utils._parse_user_dt(end, tz)
        end = pd.Timestamp.fromtimestamp(end_ts).tz_localize('UTC').tz_convert(tz)
    if end is None:
        end = dt_now
    if start is None:
        start = end - pd.Timedelta(days=548)
    if start >= end:
        logger.error('Start date must be before end')
        return None
    start = start.floor('D')
    end = end.ceil('D')
    ts_url_base = f'https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{self.ticker}?symbol={self.ticker}'
    shares_url = f'{ts_url_base}&period1={int(start.timestamp())}&period2={int(end.timestamp())}'
    try:
        json_data = self._data.cache_get(url=shares_url, proxy=proxy)
        json_data = json_data.json()
    except (_json.JSONDecodeError, requests.exceptions.RequestException):
        logger.error(f'{self.ticker}: Yahoo web request for share count failed')
        return None
    try:
        fail = json_data['finance']['error']['code'] == 'Bad Request'
    except KeyError:
        fail = False
    if fail:
        logger.error(f'{self.ticker}: Yahoo web request for share count failed')
        return None
    shares_data = json_data['timeseries']['result']
    if 'shares_out' not in shares_data[0]:
        return None
    try:
        df = pd.Series(shares_data[0]['shares_out'], index=pd.to_datetime(shares_data[0]['timestamp'], unit='s'))
    except Exception as e:
        logger.error(f'{self.ticker}: Failed to parse shares count data: {e}')
        return None
    df.index = df.index.tz_localize(tz)
    df = df.sort_index()
    return df