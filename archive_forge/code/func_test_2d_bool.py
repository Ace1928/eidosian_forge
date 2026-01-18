from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_2d_bool(self):
    arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)
    result = algos.take_nd(arr, [0, 2, 2, 1])
    expected = arr.take([0, 2, 2, 1], axis=0)
    tm.assert_numpy_array_equal(result, expected)
    result = algos.take_nd(arr, [0, 2, 2, 1], axis=1)
    expected = arr.take([0, 2, 2, 1], axis=1)
    tm.assert_numpy_array_equal(result, expected)
    result = algos.take_nd(arr, [0, 2, -1])
    assert result.dtype == np.object_