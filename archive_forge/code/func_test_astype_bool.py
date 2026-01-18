import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_astype_bool(self):
    a = SparseArray([1, 0, 0, 1], dtype=SparseDtype(int, 0))
    result = a.astype(bool)
    expected = np.array([1, 0, 0, 1], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
    result = a.astype(SparseDtype(bool, False))
    expected = SparseArray([True, False, False, True], dtype=SparseDtype(bool, False))
    tm.assert_sp_array_equal(result, expected)