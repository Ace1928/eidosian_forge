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
def test_full_groupby():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
    ddf = dd.from_pandas(df, npartitions=3)
    pytest.raises(KeyError, lambda: ddf.groupby('does_not_exist'))
    pytest.raises(AttributeError, lambda: ddf.groupby('a').does_not_exist)
    assert 'b' in dir(ddf.groupby('a'))

    def func(df):
        return df.assign(b=df.b - df.b.mean())
    expected = df.groupby('a').apply(func, **INCLUDE_GROUPS)
    with pytest.warns(UserWarning, match='`meta` is not specified'):
        if not DASK_EXPR_ENABLED:
            assert ddf.groupby('a').apply(func, **INCLUDE_GROUPS)._name.startswith('func')
        assert_eq(expected, ddf.groupby('a').apply(func, **INCLUDE_GROUPS))