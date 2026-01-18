from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="axis doesn't exist in dask-expr")
@pytest.mark.parametrize('func', ['cumsum', 'cumprod'])
def test_cumulative_axis(func):
    df = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7] * 2, 'b': np.random.randn(18), 'c': np.random.randn(18)})
    df.iloc[-6, -1] = np.nan
    ddf = dd.from_pandas(df, npartitions=4)
    expected = getattr(df.groupby('a'), func)()
    result = getattr(ddf.groupby('a'), func)()
    assert_eq(expected, result)
    axis_deprecated = contextlib.nullcontext()
    if not PANDAS_GE_210:
        axis_deprecated = pytest.warns(FutureWarning, match="'axis' keyword is deprecated")
    with groupby_axis_deprecated():
        expected = getattr(df.groupby('a'), func)(axis=0)
    with groupby_axis_deprecated(axis_deprecated):
        result = getattr(ddf.groupby('a'), func)(axis=0)
    assert_eq(expected, result)
    with groupby_axis_deprecated():
        expected = getattr(df.groupby('a'), func)(axis=1)
    with groupby_axis_deprecated(axis_deprecated):
        result = getattr(ddf.groupby('a'), func)(axis=1)
    assert_eq(expected, result)
    with groupby_axis_deprecated(pytest.raises(ValueError, match='No axis named 1 for object type Series'), axis_deprecated):
        getattr(ddf.groupby('a').b, func)(axis=1)
    with pytest.warns(FutureWarning, match="'axis' keyword is deprecated"):
        ddf.groupby('a').cumcount(axis=1)