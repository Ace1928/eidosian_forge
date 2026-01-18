from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.parametrize('left, right', [(np.array([], dtype='int64'), np.array([], dtype='int64')), (np.array([0], dtype='int64'), np.array([1], dtype='int64')), (np.array([np.nan]), np.array([np.nan])), (np.array([np.nan] * 3), np.array([np.nan] * 3))])
def test_is_overlapping_trivial(self, closed, left, right):
    tree = IntervalTree(left, right, closed=closed)
    assert tree.is_overlapping is False