from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_merge_known_to_known(df_left, df_right, ddf_left, ddf_right, on, how, shuffle_method):
    expected = df_left.merge(df_right, on=on, how=how)
    result = ddf_left.merge(ddf_right, on=on, how=how, shuffle_method=shuffle_method)
    assert_eq(result, expected)
    assert_eq(result.divisions, tuple(range(12)))
    assert len(result.__dask_graph__()) < 80