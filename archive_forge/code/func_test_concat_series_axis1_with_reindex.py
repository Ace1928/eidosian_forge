import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_series_axis1_with_reindex(self, sort):
    s = Series(np.random.default_rng(2).standard_normal(3), index=['c', 'a', 'b'], name='A')
    s2 = Series(np.random.default_rng(2).standard_normal(4), index=['d', 'a', 'b', 'c'], name='B')
    result = concat([s, s2], axis=1, sort=sort)
    expected = DataFrame({'A': s, 'B': s2}, index=['c', 'a', 'b', 'd'])
    if sort:
        expected = expected.sort_index()
    tm.assert_frame_equal(result, expected)