import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_get_indexer_backfill(self):
    index = RangeIndex(start=0, stop=20, step=2)
    target = RangeIndex(10)
    indexer = index.get_indexer(target, method='backfill')
    expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)