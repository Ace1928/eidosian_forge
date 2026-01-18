from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def safe_merge_dfs(df_main, df_sub, interval):
    if df_sub.empty:
        raise Exception('No data to merge')
    if df_main.empty:
        return df_main
    data_cols = [c for c in df_sub.columns if c not in df_main]
    if len(data_cols) > 1:
        raise Exception('Expected 1 data col')
    data_col = data_cols[0]
    df_main = df_main.sort_index()
    intraday = interval.endswith('m') or interval.endswith('s')
    td = _interval_to_timedelta(interval)
    if intraday:
        df_main['_date'] = df_main.index.date
        df_sub['_date'] = df_sub.index.date
        indices = _np.searchsorted(_np.append(df_main['_date'], [df_main['_date'].iloc[-1] + td]), df_sub['_date'], side='left')
        df_main = df_main.drop('_date', axis=1)
        df_sub = df_sub.drop('_date', axis=1)
    else:
        indices = _np.searchsorted(_np.append(df_main.index, df_main.index[-1] + td), df_sub.index, side='right')
        indices -= 1
    if intraday:
        for i in range(len(df_sub.index)):
            dt = df_sub.index[i].date()
            if dt < df_main.index[0].date() or dt >= df_main.index[-1].date() + _datetime.timedelta(days=1):
                indices[i] = -1
    else:
        for i in range(len(df_sub.index)):
            dt = df_sub.index[i]
            if dt < df_main.index[0] or dt >= df_main.index[-1] + td:
                indices[i] = -1
    f_outOfRange = indices == -1
    if f_outOfRange.any():
        if intraday:
            df_sub = df_sub[~f_outOfRange]
            if df_sub.empty:
                df_main['Dividends'] = 0.0
                return df_main
        else:
            empty_row_data = {**{c: [_np.nan] for c in const._PRICE_COLNAMES_}, 'Volume': [0]}
            if interval == '1d':
                for i in _np.where(f_outOfRange)[0]:
                    dt = df_sub.index[i]
                    get_yf_logger().debug(f'Adding out-of-range {data_col} @ {dt.date()} in new prices row of NaNs')
                    empty_row = _pd.DataFrame(data=empty_row_data, index=[dt])
                    df_main = _pd.concat([df_main, empty_row], sort=True)
            else:
                last_dt = df_main.index[-1]
                next_interval_start_dt = last_dt + td
                next_interval_end_dt = next_interval_start_dt + td
                for i in _np.where(f_outOfRange)[0]:
                    dt = df_sub.index[i]
                    if next_interval_start_dt <= dt < next_interval_end_dt:
                        get_yf_logger().debug(f'Adding out-of-range {data_col} @ {dt.date()} in new prices row of NaNs')
                        empty_row = _pd.DataFrame(data=empty_row_data, index=[dt])
                        df_main = _pd.concat([df_main, empty_row], sort=True)
            df_main = df_main.sort_index()
            indices = _np.searchsorted(_np.append(df_main.index, df_main.index[-1] + td), df_sub.index, side='right')
            indices -= 1
            for i in range(len(df_sub.index)):
                dt = df_sub.index[i]
                if dt < df_main.index[0] or dt >= df_main.index[-1] + td:
                    indices[i] = -1
    f_outOfRange = indices == -1
    if f_outOfRange.any():
        if intraday or interval in ['1d', '1wk']:
            raise Exception(f"The following '{data_col}' events are out-of-range, did not expect with interval {interval}: {df_sub.index[f_outOfRange]}")
        get_yf_logger().debug(f'Discarding these {data_col} events:' + '\n' + str(df_sub[f_outOfRange]))
        df_sub = df_sub[~f_outOfRange].copy()
        indices = indices[~f_outOfRange]

    def _reindex_events(df, new_index, data_col_name):
        if len(new_index) == len(set(new_index)):
            df.index = new_index
            return df
        df['_NewIndex'] = new_index
        if data_col_name in ['Dividends', 'Capital Gains']:
            df = df.groupby('_NewIndex').sum()
            df.index.name = None
        elif data_col_name == 'Stock Splits':
            df = df.groupby('_NewIndex').prod()
            df.index.name = None
        else:
            raise Exception(f"New index contains duplicates but unsure how to aggregate for '{data_col_name}'")
        if '_NewIndex' in df.columns:
            df = df.drop('_NewIndex', axis=1)
        return df
    new_index = df_main.index[indices]
    df_sub = _reindex_events(df_sub, new_index, data_col)
    df = df_main.join(df_sub)
    f_na = df[data_col].isna()
    data_lost = sum(~f_na) < df_sub.shape[0]
    if data_lost:
        raise Exception('Data was lost in merge, investigate')
    return df