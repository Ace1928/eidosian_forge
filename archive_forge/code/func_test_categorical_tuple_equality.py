import numpy as np
from pandas import (
import pandas._testing as tm
def test_categorical_tuple_equality(self):
    ser = Series([(0, 0), (0, 1), (0, 0), (1, 0), (1, 1)])
    expected = Series([True, False, True, False, False])
    result = ser == (0, 0)
    tm.assert_series_equal(result, expected)
    result = ser.astype('category') == (0, 0)
    tm.assert_series_equal(result, expected)