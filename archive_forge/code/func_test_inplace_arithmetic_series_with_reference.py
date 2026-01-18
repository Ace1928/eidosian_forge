import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_inplace_arithmetic_series_with_reference(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    view = ser[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        ser *= 2
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), get_array(view))
        tm.assert_series_equal(ser_orig, view)
    else:
        assert np.shares_memory(get_array(ser), get_array(view))