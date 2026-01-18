from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_1d_other_dtypes(self):
    arr = np.random.default_rng(2).standard_normal(10).astype(np.float32)
    indexer = [1, 2, 3, -1]
    result = algos.take_nd(arr, indexer)
    expected = arr.take(indexer)
    expected[-1] = np.nan
    tm.assert_almost_equal(result, expected)