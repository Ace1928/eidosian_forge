import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
@pytest.mark.parametrize('deep', ['default', None, False, True])
def test_copy_tzaware(self, deep, using_copy_on_write):
    expected = Series([Timestamp('2012/01/01', tz='UTC')])
    expected2 = Series([Timestamp('1999/01/01', tz='UTC')])
    ser = Series([Timestamp('2012/01/01', tz='UTC')])
    if deep == 'default':
        ser2 = ser.copy()
    else:
        ser2 = ser.copy(deep=deep)
    if using_copy_on_write:
        if deep is None or deep is False:
            assert np.may_share_memory(ser.values, ser2.values)
        else:
            assert not np.may_share_memory(ser.values, ser2.values)
    ser2[0] = Timestamp('1999/01/01', tz='UTC')
    if deep is not False or using_copy_on_write:
        tm.assert_series_equal(ser2, expected2)
        tm.assert_series_equal(ser, expected)
    else:
        tm.assert_series_equal(ser2, expected2)
        tm.assert_series_equal(ser, expected2)