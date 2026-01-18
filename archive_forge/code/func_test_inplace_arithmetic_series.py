import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_inplace_arithmetic_series(using_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    data = get_array(ser)
    ser *= 2
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), data)
        tm.assert_numpy_array_equal(data, get_array(ser_orig))
    else:
        assert np.shares_memory(get_array(ser), data)
        tm.assert_numpy_array_equal(data, get_array(ser))