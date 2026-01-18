import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_intersection_empty_result(self, closed, sort):
    index = monotonic_index(0, 11, closed=closed)
    other = monotonic_index(300, 314, closed=closed)
    expected = empty_index(dtype='int64', closed=closed)
    result = index.intersection(other, sort=sort)
    tm.assert_index_equal(result, expected)
    other = monotonic_index(300, 314, dtype='float64', closed=closed)
    result = index.intersection(other, sort=sort)
    expected = other[:0]
    tm.assert_index_equal(result, expected)
    other = monotonic_index(300, 314, dtype='uint64', closed=closed)
    result = index.intersection(other, sort=sort)
    tm.assert_index_equal(result, expected)