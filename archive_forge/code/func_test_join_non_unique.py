import numpy as np
from pandas import (
import pandas._testing as tm
def test_join_non_unique(self):
    index = RangeIndex(start=0, stop=20, step=2)
    other = Index([4, 4, 3, 3])
    res, lidx, ridx = index.join(other, return_indexers=True)
    eres = Index([0, 2, 4, 4, 6, 8, 10, 12, 14, 16, 18])
    elidx = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
    eridx = np.array([-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)
    tm.assert_index_equal(res, eres)
    tm.assert_numpy_array_equal(lidx, elidx)
    tm.assert_numpy_array_equal(ridx, eridx)