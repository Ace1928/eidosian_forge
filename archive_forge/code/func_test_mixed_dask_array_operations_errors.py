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
def test_mixed_dask_array_operations_errors():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=[4, 5, 6, 7, 8])
    ddf = dd.from_pandas(df, npartitions=2)
    x = da.arange(5, chunks=((1, 4),))
    x._chunks = ((np.nan, np.nan),)
    with pytest.raises(ValueError):
        (ddf.x + x).compute()
    x = da.arange(5, chunks=((2, 2, 1),))
    with pytest.raises(ValueError) as info:
        ddf.x + x
    assert 'add' in str(info.value) or 'different numbers' in str(info.value)