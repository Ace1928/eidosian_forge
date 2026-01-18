import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import (
def test_merge_cross_more_than_one_column():
    left = DataFrame({'A': list('ab'), 'B': [2, 1]})
    right = DataFrame({'C': range(2), 'D': range(4, 6)})
    result = merge(left, right, how='cross')
    expected = DataFrame({'A': ['a', 'a', 'b', 'b'], 'B': [2, 2, 1, 1], 'C': [0, 1, 0, 1], 'D': [4, 5, 4, 5]})
    tm.assert_frame_equal(result, expected)