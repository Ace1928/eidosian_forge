import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_spindex_dtype(self):
    arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]))
    expected = SparseArray([0, 1, 2, 0], kind='integer')
    tm.assert_sp_array_equal(arr, expected)
    assert arr.dtype == SparseDtype(np.int64)
    assert arr.fill_value == 0
    arr = SparseArray(data=[1, 2, 3], sparse_index=IntIndex(4, [1, 2, 3]), dtype=np.int64, fill_value=0)
    exp = SparseArray([0, 1, 2, 3], dtype=np.int64, fill_value=0)
    tm.assert_sp_array_equal(arr, exp)
    assert arr.dtype == SparseDtype(np.int64)
    assert arr.fill_value == 0
    arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=np.int64)
    exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=np.int64)
    tm.assert_sp_array_equal(arr, exp)
    assert arr.dtype == SparseDtype(np.int64)
    assert arr.fill_value == 0
    arr = SparseArray(data=[1, 2, 3], sparse_index=IntIndex(4, [1, 2, 3]), dtype=None, fill_value=0)
    exp = SparseArray([0, 1, 2, 3], dtype=None)
    tm.assert_sp_array_equal(arr, exp)
    assert arr.dtype == SparseDtype(np.int64)
    assert arr.fill_value == 0