import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('method,expected', [('pad', [8, 7, 0]), ('backfill', [9, 8, 1]), ('nearest', [9, 7, 0])])
def test_get_indexer_nearest_decreasing(self, method, expected):
    index = Index(np.arange(10))[::-1]
    actual = index.get_indexer([0, 5, 9], method=method)
    tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))
    actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
    tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))