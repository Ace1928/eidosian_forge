import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import SparseArray
@pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
def test_abs_operator(self):
    arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
    res = abs(arr)
    exp = SparseArray([1, 2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
    tm.assert_sp_array_equal(exp, res)
    arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)
    res = abs(arr)
    exp = SparseArray([1, 2, 1, 3], fill_value=1, dtype=np.int8)
    tm.assert_sp_array_equal(exp, res)