import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('format', ['coo', 'csc', 'csr'])
def test_from_spmatrix_including_explicit_zero(self, format):
    sp_sparse = pytest.importorskip('scipy.sparse')
    mat = sp_sparse.random(10, 1, density=0.5, format=format)
    mat.data[0] = 0
    result = SparseArray.from_spmatrix(mat)
    result = np.asarray(result)
    expected = mat.toarray().ravel()
    tm.assert_numpy_array_equal(result, expected)