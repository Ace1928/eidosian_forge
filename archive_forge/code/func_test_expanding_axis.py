import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_axis(axis_frame):
    df = DataFrame(np.ones((10, 20)))
    axis = df._get_axis_number(axis_frame)
    if axis == 0:
        msg = "The 'axis' keyword in DataFrame.expanding is deprecated"
        expected = DataFrame({i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)})
    else:
        msg = 'Support for axis=1 in DataFrame.expanding is deprecated'
        expected = DataFrame([[np.nan] * 2 + [float(i) for i in range(3, 21)]] * 10)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.expanding(3, axis=axis_frame).sum()
    tm.assert_frame_equal(result, expected)