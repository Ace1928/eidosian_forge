import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
def get_financials_time_series(self, timescale, keys: list, proxy=None) -> pd.DataFrame:
    timescale_translation = {'yearly': 'annual', 'quarterly': 'quarterly'}
    timescale = timescale_translation[timescale]
    ts_url_base = f'https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{self._symbol}?symbol={self._symbol}'
    url = ts_url_base + '&type=' + ','.join([timescale + k for k in keys])
    start_dt = datetime.datetime(2016, 12, 31)
    end = pd.Timestamp.utcnow().ceil('D')
    url += f'&period1={int(start_dt.timestamp())}&period2={int(end.timestamp())}'
    json_str = self._data.cache_get(url=url, proxy=proxy).text
    json_data = json.loads(json_str)
    data_raw = json_data['timeseries']['result']
    for d in data_raw:
        del d['meta']
    timestamps = set()
    data_unpacked = {}
    for x in data_raw:
        for k in x.keys():
            if k == 'timestamp':
                timestamps.update(x[k])
            else:
                data_unpacked[k] = x[k]
    timestamps = sorted(list(timestamps))
    dates = pd.to_datetime(timestamps, unit='s')
    df = pd.DataFrame(columns=dates, index=list(data_unpacked.keys()))
    for k, v in data_unpacked.items():
        if df is None:
            df = pd.DataFrame(columns=dates, index=[k])
        df.loc[k] = {pd.Timestamp(x['asOfDate']): x['reportedValue']['raw'] for x in v}
    df.index = df.index.str.replace('^' + timescale, '', regex=True)
    df = df.reindex([k for k in keys if k in df.index])
    df = df[sorted(df.columns, reverse=True)]
    return df