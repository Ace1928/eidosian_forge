import numpy as np
from pandas import (
import pandas._testing as tm
def test_join_outer(self):
    index = RangeIndex(start=0, stop=20, step=2)
    other = Index(np.arange(25, 14, -1, dtype=np.int64))
    res, lidx, ridx = index.join(other, how='outer', return_indexers=True)
    noidx_res = index.join(other, how='outer')
    tm.assert_index_equal(res, noidx_res)
    eres = Index([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    elidx = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, 8, -1, 9, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)
    eridx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.intp)
    assert isinstance(res, Index) and res.dtype == np.dtype(np.int64)
    assert not isinstance(res, RangeIndex)
    tm.assert_index_equal(res, eres, exact=True)
    tm.assert_numpy_array_equal(lidx, elidx)
    tm.assert_numpy_array_equal(ridx, eridx)
    other = RangeIndex(25, 14, -1)
    res, lidx, ridx = index.join(other, how='outer', return_indexers=True)
    noidx_res = index.join(other, how='outer')
    tm.assert_index_equal(res, noidx_res)
    assert isinstance(res, Index) and res.dtype == np.int64
    assert not isinstance(res, RangeIndex)
    tm.assert_index_equal(res, eres)
    tm.assert_numpy_array_equal(lidx, elidx)
    tm.assert_numpy_array_equal(ridx, eridx)