import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_floordiv_by_int_zero_no_mask(any_int_ea_dtype):
    ser = pd.Series([0, 1], dtype=any_int_ea_dtype)
    result = 1 // ser
    expected = pd.Series([np.inf, 1.0], dtype='Float64')
    tm.assert_series_equal(result, expected)
    ser_non_nullable = ser.astype(ser.dtype.numpy_dtype)
    result = 1 // ser_non_nullable
    expected = expected.astype(np.float64)
    tm.assert_series_equal(result, expected)