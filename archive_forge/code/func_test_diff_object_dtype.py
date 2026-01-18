import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_diff_object_dtype(self):
    ser = Series([False, True, 5.0, np.nan, True, False])
    result = ser.diff()
    expected = ser - ser.shift(1)
    tm.assert_series_equal(result, expected)