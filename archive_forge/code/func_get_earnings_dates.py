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
def get_earnings_dates(self, limit=12, proxy=None) -> Optional[pd.DataFrame]:
    """
        Get earning dates (future and historic)
        :param limit: max amount of upcoming and recent earnings dates to return.
                      Default value 12 should return next 4 quarters and last 8 quarters.
                      Increase if more history is needed.

        :param proxy: requests proxy to use.
        :return: pandas dataframe
        """
    if self._earnings_dates and limit in self._earnings_dates:
        return self._earnings_dates[limit]
    logger = utils.get_yf_logger()
    page_size = min(limit, 100)
    page_offset = 0
    dates = None
    while True:
        url = f'{_ROOT_URL_}/calendar/earnings?symbol={self.ticker}&offset={page_offset}&size={page_size}'
        data = self._data.cache_get(url=url, proxy=proxy).text
        if 'Will be right back' in data:
            raise RuntimeError('*** YAHOO! FINANCE IS CURRENTLY DOWN! ***\nOur engineers are working quickly to resolve the issue. Thank you for your patience.')
        try:
            data = pd.read_html(StringIO(data))[0]
        except ValueError:
            if page_offset == 0:
                if 'Showing Earnings for:' in data:
                    dates = utils.empty_earnings_dates_df()
            break
        if dates is None:
            dates = data
        else:
            dates = pd.concat([dates, data], axis=0)
        page_offset += page_size
        if len(data) < page_size or len(dates) >= limit:
            dates = dates.iloc[:limit]
            break
        else:
            page_size = min(limit - len(dates), page_size)
    if dates is None or dates.shape[0] == 0:
        err_msg = 'No earnings dates found, symbol may be delisted'
        logger.error(f'{self.ticker}: {err_msg}')
        return None
    dates = dates.reset_index(drop=True)
    dates = dates.drop(['Symbol', 'Company'], axis=1)
    for cn in ['EPS Estimate', 'Reported EPS', 'Surprise(%)']:
        dates.loc[dates[cn] == '-', cn] = float('nan')
        dates[cn] = dates[cn].astype(float)
    dates['Surprise(%)'] *= 0.01
    cn = 'Earnings Date'
    tzinfo = dates[cn].str.extract('([AP]M[a-zA-Z]*)$')
    dates[cn] = dates[cn].replace(' [AP]M[a-zA-Z]*$', '', regex=True)
    tzinfo = tzinfo[0].str.extract('([AP]M)([a-zA-Z]*)', expand=True)
    tzinfo.columns = ['AM/PM', 'TZ']
    dates[cn] = dates[cn] + ' ' + tzinfo['AM/PM']
    dates[cn] = pd.to_datetime(dates[cn], format='%b %d, %Y, %I %p')
    self._quote.proxy = proxy or self.proxy
    tz = self._get_ticker_tz(proxy=proxy, timeout=30)
    dates[cn] = dates[cn].dt.tz_localize(tz)
    dates = dates.set_index('Earnings Date')
    self._earnings_dates[limit] = dates
    return dates