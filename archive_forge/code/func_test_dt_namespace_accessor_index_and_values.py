import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dt_namespace_accessor_index_and_values(self):
    index = date_range('20130101', periods=3, freq='D')
    dti = date_range('20140204', periods=3, freq='s')
    ser = Series(dti, index=index, name='xxx')
    exp = Series(np.array([2014, 2014, 2014], dtype='int32'), index=index, name='xxx')
    tm.assert_series_equal(ser.dt.year, exp)
    exp = Series(np.array([2, 2, 2], dtype='int32'), index=index, name='xxx')
    tm.assert_series_equal(ser.dt.month, exp)
    exp = Series(np.array([0, 1, 2], dtype='int32'), index=index, name='xxx')
    tm.assert_series_equal(ser.dt.second, exp)
    exp = Series([ser.iloc[0]] * 3, index=index, name='xxx')
    tm.assert_series_equal(ser.dt.normalize(), exp)