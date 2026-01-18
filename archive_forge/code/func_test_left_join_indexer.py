import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
def test_left_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)
    index, ares, bres = libjoin.left_join_indexer(a, b)
    tm.assert_almost_equal(index, a)
    aexp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
    bexp = np.array([-1, -1, 1, -1, 2], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)
    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)
    index, ares, bres = libjoin.left_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))