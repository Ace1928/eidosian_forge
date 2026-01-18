import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_isetitem_series(using_copy_on_write, dtype):
    df = DataFrame({'a': [1, 2, 3], 'b': np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    ser_orig = ser.copy()
    df.isetitem(0, ser)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'a'), get_array(ser))
        assert not df._mgr._has_no_reference(0)
    df.loc[0, 'a'] = 0
    tm.assert_series_equal(ser, ser_orig)
    df = DataFrame({'a': [1, 2, 3], 'b': np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    df.isetitem(0, ser)
    ser.loc[0] = 0
    expected = DataFrame({'a': [7, 8, 9], 'b': np.array([4, 5, 6], dtype=dtype)})
    tm.assert_frame_equal(df, expected)