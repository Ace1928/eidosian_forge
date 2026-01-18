import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_sparse_dtype(self):
    result = SparseArray([1, 0, 0, 1], dtype=SparseDtype('int64', -1))
    expected = SparseArray([1, 0, 0, 1], fill_value=-1, dtype=np.int64)
    tm.assert_sp_array_equal(result, expected)
    assert result.sp_values.dtype == np.dtype('int64')