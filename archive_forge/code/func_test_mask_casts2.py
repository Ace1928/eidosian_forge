import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_mask_casts2():
    ser = Series([1, 2])
    res = ser.mask([True, False])
    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)