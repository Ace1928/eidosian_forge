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
def test_groupby_error():
    pdf = pd.DataFrame({'x': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], 'y': list('abcbabbcda')})
    ddf = dd.from_pandas(pdf, 3)
    with pytest.raises(KeyError):
        ddf.groupby('A')
    with pytest.raises(KeyError):
        ddf.groupby(['x', 'A'])
    dp = ddf.groupby('y')
    msg = 'Column not found: '
    with pytest.raises(KeyError) as err:
        dp['A']
    assert msg in str(err.value)
    msg = 'Columns not found: '
    with pytest.raises(KeyError) as err:
        dp[['x', 'A']]
    assert msg in str(err.value)
    msg = 'DataFrameGroupBy does not allow compute method.Please chain it with an aggregation method (like ``.mean()``) or get a specific group using ``.get_group()`` before calling ``compute()``'
    with pytest.raises(NotImplementedError) as err:
        dp.compute()
    assert msg in str(err.value)