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
def test_idxmaxmin_empty_partitions():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2, 3], 'c': [np.nan] * 3, 'd': [1, 2, np.nan]})
    empty = df.iloc[:0]
    ddf = dd.concat([dd.from_pandas(df, npartitions=1)] + [dd.from_pandas(empty, npartitions=1)] * 10)
    if PANDAS_GE_300:
        ctx = pytest.raises(ValueError, match='Encountered all NA values')
    elif PANDAS_GE_210:
        ctx = pytest.warns(FutureWarning, match='all-NA values')
    else:
        ctx = contextlib.nullcontext()
    for skipna in [True, False]:
        with ctx:
            expected = df.idxmin(skipna=skipna)
        if not PANDAS_GE_300:
            result = ddf.idxmin(skipna=skipna, split_every=3)
            with ctx:
                assert_eq(result, expected)
    assert_eq(ddf[['a', 'b', 'd']].idxmin(skipna=True, split_every=3), df[['a', 'b', 'd']].idxmin(skipna=True))
    assert_eq(ddf.b.idxmax(split_every=3), df.b.idxmax())
    ddf = dd.concat([dd.from_pandas(empty, npartitions=1)] * 10)
    with pytest.raises(ValueError):
        ddf.idxmax().compute()
    with pytest.raises(ValueError):
        ddf.b.idxmax().compute()