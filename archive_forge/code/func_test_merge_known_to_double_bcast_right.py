from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('how', ['inner', 'left'])
def test_merge_known_to_double_bcast_right(df_left, df_right, ddf_left, ddf_right_double, on, how, shuffle_method):
    expected = df_left.merge(df_right, on=on, how=how)
    result = ddf_left.merge(ddf_right_double, on=on, how=how, shuffle_method=shuffle_method, broadcast=True)
    assert_eq(result, expected)
    if shuffle_method == 'task':
        assert_eq(result.divisions, ddf_left.divisions)