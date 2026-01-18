import numpy as np
from pandas import (
def test_get_dummies_with_name_dummy(any_string_dtype):
    s = Series(['a', 'b,name', 'b'], dtype=any_string_dtype)
    result = s.str.get_dummies(',')
    expected = DataFrame([[1, 0, 0], [0, 1, 1], [0, 1, 0]], columns=['a', 'b', 'name'])
    tm.assert_frame_equal(result, expected)