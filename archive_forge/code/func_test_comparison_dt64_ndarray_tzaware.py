from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('reverse', [True, False])
def test_comparison_dt64_ndarray_tzaware(self, reverse, comparison_op):
    ts = Timestamp('2021-01-01 00:00:00.00000', tz='UTC')
    arr = np.array([ts.asm8, ts.asm8], dtype='M8[ns]')
    left, right = (ts, arr)
    if reverse:
        left, right = (arr, ts)
    if comparison_op is operator.eq:
        expected = np.array([False, False], dtype=bool)
        result = comparison_op(left, right)
        tm.assert_numpy_array_equal(result, expected)
    elif comparison_op is operator.ne:
        expected = np.array([True, True], dtype=bool)
        result = comparison_op(left, right)
        tm.assert_numpy_array_equal(result, expected)
    else:
        msg = 'Cannot compare tz-naive and tz-aware timestamps'
        with pytest.raises(TypeError, match=msg):
            comparison_op(left, right)