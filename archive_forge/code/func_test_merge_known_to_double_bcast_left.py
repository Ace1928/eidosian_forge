from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('how', ['inner', 'right'])
@pytest.mark.parametrize('broadcast', [True, 0.75])
def test_merge_known_to_double_bcast_left(df_left, df_right, ddf_left_double, ddf_right, on, shuffle_method, how, broadcast):
    expected = df_left.merge(df_right, on=on, how=how)
    result = ddf_left_double.merge(ddf_right, on=on, how=how, broadcast=broadcast, shuffle_method=shuffle_method)
    assert_eq(result, expected)
    if shuffle_method == 'task':
        assert_eq(result.divisions, ddf_right.divisions)
    result.head(1)