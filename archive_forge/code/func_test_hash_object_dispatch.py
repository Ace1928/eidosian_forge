from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.util import hash_pandas_object
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import tm
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('obj', [pd.Index([1, 2, 3]), pd.Index([True, False, True]), pd.Series([1, 2, 3]), pd.Series([1.0, 1.5, 3.2]), pd.Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]), pd.DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]}), pd.DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]}, index=['a', 'z', 'x'])])
def test_hash_object_dispatch(obj):
    result = dd.dispatch.hash_object_dispatch(obj)
    expected = pd.util.hash_pandas_object(obj)
    assert_eq(result, expected)