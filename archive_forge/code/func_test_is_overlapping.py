from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.parametrize('left, right, expected', [(np.array([0, 1, 4], dtype='int64'), np.array([2, 3, 5]), True), (np.array([0, 1, 2], dtype='int64'), np.array([5, 4, 3]), True), (np.array([0, 1, np.nan]), np.array([5, 4, np.nan]), True), (np.array([0, 2, 4], dtype='int64'), np.array([1, 3, 5]), False), (np.array([0, 2, np.nan]), np.array([1, 3, np.nan]), False)])
@pytest.mark.parametrize('order', (list(x) for x in permutations(range(3))))
def test_is_overlapping(self, closed, order, left, right, expected):
    tree = IntervalTree(left[order], right[order], closed=closed)
    result = tree.is_overlapping
    assert result is expected