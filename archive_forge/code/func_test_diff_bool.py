import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input,output,diff', [([False, True, True, False, False], [np.nan, True, False, True, False], 1)])
def test_diff_bool(self, input, output, diff):
    ser = Series(input)
    result = ser.diff()
    expected = Series(output)
    tm.assert_series_equal(result, expected)