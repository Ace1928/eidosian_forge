import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_series_axis1_names_applied(self):
    s = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = concat([s, s2], axis=1, keys=['a', 'b'], names=['A'])
    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=Index(['a', 'b'], name='A'))
    tm.assert_frame_equal(result, expected)
    result = concat([s, s2], axis=1, keys=[('a', 1), ('b', 2)], names=['A', 'B'])
    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['A', 'B']))
    tm.assert_frame_equal(result, expected)