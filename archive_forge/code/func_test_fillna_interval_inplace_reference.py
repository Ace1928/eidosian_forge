import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_interval_inplace_reference(using_copy_on_write, warn_copy_on_write):
    ser = Series(interval_range(start=0, end=5), name='a', dtype='interval[float64, right]')
    ser.iloc[1] = np.nan
    ser_orig = ser.copy()
    view = ser[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.fillna(value=Interval(left=0, right=5), inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser, 'a').left.values, get_array(view, 'a').left.values)
        tm.assert_series_equal(view, ser_orig)
    else:
        assert np.shares_memory(get_array(ser, 'a').left.values, get_array(view, 'a').left.values)