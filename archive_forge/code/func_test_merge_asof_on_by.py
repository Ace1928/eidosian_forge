from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
def test_merge_asof_on_by():
    times_A = [pd.to_datetime(d) for d in ['2016-05-25 13:30:00.023', '2016-05-25 13:30:00.023', '2016-05-25 13:30:00.030', '2016-05-25 13:30:00.041', '2016-05-25 13:30:00.048', '2016-05-25 13:30:00.049', '2016-05-25 13:30:00.072', '2016-05-25 13:30:00.075']]
    tickers_A = ['GOOG', 'MSFT', 'MSFT', 'MSFT', 'GOOG', 'AAPL', 'GOOG', 'MSFT']
    bids_A = [720.5, 51.95, 51.97, 51.99, 720.5, 97.99, 720.5, 52.01]
    asks_A = [720.93, 51.96, 51.98, 52.0, 720.93, 98.01, 720.88, 52.03]
    times_B = [pd.to_datetime(d) for d in ['2016-05-25 13:30:00.023', '2016-05-25 13:30:00.038', '2016-05-25 13:30:00.048', '2016-05-25 13:30:00.048', '2016-05-25 13:30:00.048']]
    tickers_B = ['MSFT', 'MSFT', 'GOOG', 'GOOG', 'AAPL']
    prices_B = [51.95, 51.95, 720.77, 720.92, 98.0]
    quantities_B = [75, 155, 100, 100, 100]
    A = pd.DataFrame({'time': times_A, 'ticker': tickers_A, 'bid': bids_A, 'ask': asks_A}, columns=['time', 'ticker', 'bid', 'ask'])
    a = dd.from_pandas(A, npartitions=4)
    B = pd.DataFrame({'time': times_B, 'ticker': tickers_B, 'price': prices_B, 'quantity': quantities_B}, columns=['time', 'ticker', 'price', 'quantity'])
    b = dd.from_map(lambda x: x, [B.iloc[0:2], B.iloc[2:5]], divisions=[0, 2, 4])
    C = pd.merge_asof(B, A, on='time', by='ticker')
    c = dd.merge_asof(b, a, on='time', by='ticker')
    assert_eq(c, C, check_index=False)