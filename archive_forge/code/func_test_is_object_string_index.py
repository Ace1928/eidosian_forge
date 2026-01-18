from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.decimal.array import DecimalDtype
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_150
from dask.dataframe._pyarrow import (
@pytest.mark.parametrize('index,expected', [(pd.Index(['a', 'b'], dtype=object), True), (pd.Index(['a', 'b'], dtype='string[python]'), True), (pd.Index(['a', 'b'], dtype='string[pyarrow]'), False if PANDAS_GE_140 else True), (pd.Index([1, 2], dtype=int), False), (pd.Index([1, 2], dtype=float), False), (pd.Series(['a', 'b'], dtype=object), False), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype='string[pyarrow]'), pd.Index(['a', 'b'], dtype=object)]), True), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype='string[pyarrow]'), pd.Index(['a', 'b'], dtype='string[pyarrow]')]), False if PANDAS_GE_140 else True), (pd.MultiIndex.from_arrays([pd.Index(['a', 'a'], dtype=object), pd.Index([1, 2], dtype=int)]), True), (pd.MultiIndex.from_arrays([pd.Index([1, 1], dtype=int), pd.Index([1, 2], dtype=float)]), False)])
def test_is_object_string_index(index, expected):
    assert is_object_string_index(index) is expected