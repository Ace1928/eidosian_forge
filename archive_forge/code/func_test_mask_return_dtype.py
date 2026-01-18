import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_return_dtype():
    ser = Series([0.0, 1.0, 2.0, 3.0], dtype=Float64Dtype())
    cond = ~ser.isna()
    other = Series([True, False, True, False])
    excepted = Series([1.0, 0.0, 1.0, 0.0], dtype=ser.dtype)
    result = ser.mask(cond, other)
    tm.assert_series_equal(result, excepted)