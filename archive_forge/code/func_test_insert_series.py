import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_insert_series(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    df.insert(loc=1, value=ser, column='b')
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(df, 'b'))
        assert not df._mgr._has_no_reference(1)
    else:
        assert not np.shares_memory(get_array(ser), get_array(df, 'b'))
    df.iloc[0, 1] = 100
    tm.assert_series_equal(ser, ser_orig)