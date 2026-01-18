from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_2d_datetime64(self):
    arr = np.random.default_rng(2).integers(11045376, 11360736, (5, 3)) * 100000000000
    arr = arr.view(dtype='datetime64[ns]')
    indexer = [0, 2, -1, 1, -1]
    result = algos.take_nd(arr, indexer, axis=0)
    expected = arr.take(indexer, axis=0)
    expected.view(np.int64)[[2, 4], :] = iNaT
    tm.assert_almost_equal(result, expected)
    result = algos.take_nd(arr, indexer, axis=0, fill_value=datetime(2007, 1, 1))
    expected = arr.take(indexer, axis=0)
    expected[[2, 4], :] = datetime(2007, 1, 1)
    tm.assert_almost_equal(result, expected)
    result = algos.take_nd(arr, indexer, axis=1)
    expected = arr.take(indexer, axis=1)
    expected.view(np.int64)[:, [2, 4]] = iNaT
    tm.assert_almost_equal(result, expected)
    result = algos.take_nd(arr, indexer, axis=1, fill_value=datetime(2007, 1, 1))
    expected = arr.take(indexer, axis=1)
    expected[:, [2, 4]] = datetime(2007, 1, 1)
    tm.assert_almost_equal(result, expected)