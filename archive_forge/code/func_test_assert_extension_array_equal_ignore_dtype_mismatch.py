import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('right_dtype', ['Int32', 'int64'])
def test_assert_extension_array_equal_ignore_dtype_mismatch(right_dtype):
    left = array([1, 2, 3], dtype='Int64')
    right = array([1, 2, 3], dtype=right_dtype)
    tm.assert_extension_array_equal(left, right, check_dtype=False)