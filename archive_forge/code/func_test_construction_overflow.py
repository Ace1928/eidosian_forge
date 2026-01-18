from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.skipif(not IS64, reason='GH 23440')
def test_construction_overflow(self):
    left, right = (np.arange(101, dtype='int64'), [np.iinfo(np.int64).max] * 101)
    tree = IntervalTree(left, right)
    result = tree.root.pivot
    expected = (50 + np.iinfo(np.int64).max) / 2
    assert result == expected