from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.parametrize('dtype, target_value, target_dtype', [('int64', 2 ** 63 + 1, 'uint64'), ('uint64', -1, 'int64')])
def test_get_indexer_overflow(self, dtype, target_value, target_dtype):
    left, right = (np.array([0, 1], dtype=dtype), np.array([1, 2], dtype=dtype))
    tree = IntervalTree(left, right)
    result = tree.get_indexer(np.array([target_value], dtype=target_dtype))
    expected = np.array([-1], dtype='intp')
    tm.assert_numpy_array_equal(result, expected)