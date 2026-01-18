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
def test_clip_axis_1():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [3, 5, 2, 5, 7, 2, 4, 2, 4]})
    ddf = dd.from_pandas(df, 3)
    l = pd.Series({'a': 2, 'b': 3})
    u = pd.Series({'a': 7, 'b': 5})
    assert_eq(ddf.clip(lower=l, upper=u, axis=1), df.clip(lower=l, upper=u, axis=1))
    assert_eq(ddf.clip(lower=l, axis=1), df.clip(lower=l, axis=1))
    assert_eq(ddf.clip(upper=u, axis=1), df.clip(upper=u, axis=1))
    if DASK_EXPR_ENABLED:
        with pytest.raises(ValueError, match='No axis named 1 for Series'):
            ddf.a.clip(lower=l, upper=u, axis=1)
    else:
        with pytest.raises(ValueError, match='Series.clip does not support axis=1'):
            ddf.a.clip(lower=l, upper=u, axis=1)