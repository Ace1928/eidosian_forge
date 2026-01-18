from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_backfill(self):
    old = np.array([1, 5, 10], dtype=np.int64)
    new = np.array(list(range(12)), dtype=np.int64)
    filler = libalgos.backfill['int64_t'](old, new)
    expect_filler = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(filler, expect_filler)
    old = np.array([1, 4], dtype=np.int64)
    new = np.array(list(range(5, 10)), dtype=np.int64)
    filler = libalgos.backfill['int64_t'](old, new)
    expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(filler, expect_filler)