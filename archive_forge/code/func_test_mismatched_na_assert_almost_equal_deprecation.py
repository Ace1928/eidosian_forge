import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left', objs)
@pytest.mark.parametrize('right', objs)
def test_mismatched_na_assert_almost_equal_deprecation(left, right):
    left_arr = np.array([left], dtype=object)
    right_arr = np.array([right], dtype=object)
    msg = 'Mismatched null-like values'
    if left is right:
        _assert_almost_equal_both(left, right, check_dtype=False)
        tm.assert_numpy_array_equal(left_arr, right_arr)
        tm.assert_index_equal(Index(left_arr, dtype=object), Index(right_arr, dtype=object))
        tm.assert_series_equal(Series(left_arr, dtype=object), Series(right_arr, dtype=object))
        tm.assert_frame_equal(DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object))
    else:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            _assert_almost_equal_both(left, right, check_dtype=False)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(Series(left_arr, dtype=object), Series(right_arr, dtype=object))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object))