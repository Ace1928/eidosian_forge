import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('arr, dtype, expected', [(SparseArray([0, 1]), 'float', SparseArray([0.0, 1.0], dtype=SparseDtype(float, 0.0))), (SparseArray([0, 1]), bool, SparseArray([False, True])), (SparseArray([0, 1], fill_value=1), bool, SparseArray([False, True], dtype=SparseDtype(bool, True))), pytest.param(SparseArray([0, 1]), 'datetime64[ns]', SparseArray(np.array([0, 1], dtype='datetime64[ns]'), dtype=SparseDtype('datetime64[ns]', Timestamp('1970')))), (SparseArray([0, 1, 10]), str, SparseArray(['0', '1', '10'], dtype=SparseDtype(str, '0'))), (SparseArray(['10', '20']), float, SparseArray([10.0, 20.0])), (SparseArray([0, 1, 0]), object, SparseArray([0, 1, 0], dtype=SparseDtype(object, 0)))])
def test_astype_more(self, arr, dtype, expected):
    result = arr.astype(arr.dtype.update_dtype(dtype))
    tm.assert_sp_array_equal(result, expected)