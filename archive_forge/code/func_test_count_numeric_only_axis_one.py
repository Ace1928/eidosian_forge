from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
def test_count_numeric_only_axis_one():
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2011, i, 1) for i in range(1, 8)], 'str': list('abcdefgh'), 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, np.nan]), 'bool': [True, False] * 4})
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(ddf.count(axis=1), df.count(axis=1))
    assert_eq(ddf.count(numeric_only=False, axis=1), df.count(numeric_only=False, axis=1))
    assert_eq(ddf.count(numeric_only=True, axis=1), df.count(numeric_only=True, axis=1))