import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_from_coo(self):
    scipy_sparse = pytest.importorskip('scipy.sparse')
    row = [0, 3, 1, 0]
    col = [0, 3, 1, 2]
    data = [4, 5, 7, 9]
    sp_array = scipy_sparse.coo_matrix((data, (row, col)))
    result = pd.Series.sparse.from_coo(sp_array)
    index = pd.MultiIndex.from_arrays([np.array([0, 0, 1, 3], dtype=np.int32), np.array([0, 2, 1, 3], dtype=np.int32)])
    expected = pd.Series([4, 9, 7, 5], index=index, dtype='Sparse[int]')
    tm.assert_series_equal(result, expected)