from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.decimal.array import DecimalDtype
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_150
from dask.dataframe._pyarrow import (
@pytest.mark.parametrize('series,expected', [(pd.Series(['a', 'b'], dtype=object), True), (pd.Series(['a', 'b'], dtype='string[python]'), True), (pd.Series(['a', 'b'], dtype='string[pyarrow]'), False), (pd.Series([1, 2], dtype=int), False), (pd.Series([1, 2], dtype=float), False), (pd.Series([1, 2], dtype=float, index=pd.Index(['a', 'b'], dtype=object)), True), (pd.Series([1, 2], dtype=float, index=pd.Index(['a', 'b'], dtype='string[pyarrow]')), False if PANDAS_GE_140 else True), (pd.Index(['a', 'b'], dtype=object), False)])
def test_is_object_string_series(series, expected):
    assert is_object_string_series(series) is expected