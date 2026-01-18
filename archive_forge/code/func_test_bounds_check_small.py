from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_bounds_check_small(self):
    arr = np.array([1, 2, 3], dtype=np.int64)
    indexer = [0, -1, -2]
    msg = "'indices' contains values less than allowed \\(-2 < -1\\)"
    with pytest.raises(ValueError, match=msg):
        algos.take(arr, indexer, allow_fill=True)
    result = algos.take(arr, indexer)
    expected = np.array([1, 3, 2], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)