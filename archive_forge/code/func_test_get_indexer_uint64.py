import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_indexer_uint64(self, index_large):
    target = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
    indexer = index_large.get_indexer(target)
    expected = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)
    target = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
    indexer = index_large.get_indexer(target, method='pad')
    expected = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)
    target = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
    indexer = index_large.get_indexer(target, method='backfill')
    expected = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)