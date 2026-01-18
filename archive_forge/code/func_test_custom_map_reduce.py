from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='reduction not available')
def test_custom_map_reduce():
    df = pd.DataFrame(columns=['a'], data=[[2], [4], [8]], index=[0, 1, 2])
    ddf = dd.from_pandas(df, npartitions=2)

    def map_fn(x):
        return {'x': x, 'y': x}

    def reduce_fn(series):
        merged = None
        for mapped in series:
            if merged is None:
                merged = mapped.copy()
            else:
                merged['x'] += mapped['x']
                merged['y'] *= mapped['y']
        return merged
    string_dtype = get_string_dtype()
    result = ddf['a'].map(map_fn, meta=('data', string_dtype)).reduction(reduce_fn, aggregate=reduce_fn, meta=('data', string_dtype)).compute()[0]
    assert result == {'x': 14, 'y': 64}