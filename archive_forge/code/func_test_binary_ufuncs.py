import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('a, b', [(SparseArray([0, 0, 0]), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2]))])
@pytest.mark.parametrize('ufunc', [np.add, np.greater])
def test_binary_ufuncs(ufunc, a, b):
    result = ufunc(a, b)
    expected = ufunc(np.asarray(a), np.asarray(b))
    assert isinstance(result, SparseArray)
    tm.assert_numpy_array_equal(np.asarray(result), expected)