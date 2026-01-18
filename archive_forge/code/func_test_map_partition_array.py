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
@pytest.mark.parametrize('func', [np.asarray, M.to_records])
def test_map_partition_array(func):
    from dask.array.utils import assert_eq
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6.0, 7.0, 8.0, 9.0, 10.0]}, index=['a', 'b', 'c', 'd', 'e'])
    ddf = dd.from_pandas(df, npartitions=2)
    for pre in [lambda a: a, lambda a: a.x, lambda a: a.y, lambda a: a.index]:
        try:
            expected = func(pre(df))
        except Exception:
            continue
        x = pre(ddf).map_partitions(func)
        assert_eq(x, expected, check_type=False)
        assert isinstance(x, da.Array)
        assert x.chunks[0] == (np.nan, np.nan)