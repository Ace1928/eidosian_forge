import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_get_indexer_limit(self):
    idx = RangeIndex(4)
    target = RangeIndex(6)
    result = idx.get_indexer(target, method='pad', limit=1)
    expected = np.array([0, 1, 2, 3, 3, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)