import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ser,expected,min_periods', [(Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1), (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2), (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2), (Series([], dtype='int64'), [], 2)])
def test_iter_expanding_series(ser, expected, min_periods):
    expected = [Series(values, index=index) for values, index in expected]
    for expected, actual in zip(expected, ser.expanding(min_periods)):
        tm.assert_series_equal(actual, expected)