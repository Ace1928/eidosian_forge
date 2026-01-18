from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('how', ['inner', 'left'])
def test_merge_known_to_single(df_left, df_right, ddf_left, ddf_right_single, on, how, shuffle_method):
    expected = df_left.merge(df_right, on=on, how=how)
    result = ddf_left.merge(ddf_right_single, on=on, how=how, shuffle_method=shuffle_method)
    assert_eq(result, expected)
    assert result.divisions == ddf_left.divisions
    assert len(result.__dask_graph__()) < 30