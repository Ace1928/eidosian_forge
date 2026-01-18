import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import (
@pytest.mark.parametrize(('input_col', 'output_cols'), [('b', ['a', 'b']), ('a', ['a_x', 'a_y'])])
def test_merge_cross(input_col, output_cols):
    left = DataFrame({'a': [1, 3]})
    right = DataFrame({input_col: [3, 4]})
    left_copy = left.copy()
    right_copy = right.copy()
    result = merge(left, right, how='cross')
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(left, left_copy)
    tm.assert_frame_equal(right, right_copy)