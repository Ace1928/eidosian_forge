import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_indexer_categorical_with_nans(self):
    ii = IntervalIndex.from_breaks(range(5))
    ii2 = ii.append(IntervalIndex([np.nan]))
    ci2 = CategoricalIndex(ii2)
    result = ii2.get_indexer(ci2)
    expected = np.arange(5, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = ii2[1:].get_indexer(ci2[::-1])
    expected = np.array([3, 2, 1, 0, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    result = ii2.get_indexer(ci2.append(ci2))
    expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)