import numpy as np
from pandas import (
import pandas._testing as tm
def test_join_non_int_index(self):
    index = RangeIndex(start=0, stop=20, step=2)
    other = Index([3, 6, 7, 8, 10], dtype=object)
    outer = index.join(other, how='outer')
    outer2 = other.join(index, how='outer')
    expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
    tm.assert_index_equal(outer, outer2)
    tm.assert_index_equal(outer, expected)
    inner = index.join(other, how='inner')
    inner2 = other.join(index, how='inner')
    expected = Index([6, 8, 10])
    tm.assert_index_equal(inner, inner2)
    tm.assert_index_equal(inner, expected)
    left = index.join(other, how='left')
    tm.assert_index_equal(left, index.astype(object))
    left2 = other.join(index, how='left')
    tm.assert_index_equal(left2, other)
    right = index.join(other, how='right')
    tm.assert_index_equal(right, other)
    right2 = other.join(index, how='right')
    tm.assert_index_equal(right2, index.astype(object))