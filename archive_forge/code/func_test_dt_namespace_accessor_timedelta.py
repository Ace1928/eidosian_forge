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
def test_dt_namespace_accessor_timedelta(self):
    cases = [Series(timedelta_range('1 day', periods=5), index=list('abcde'), name='xxx'), Series(timedelta_range('1 day 01:23:45', periods=5, freq='s'), name='xxx'), Series(timedelta_range('2 days 01:23:45.012345', periods=5, freq='ms'), name='xxx')]
    for ser in cases:
        for prop in ok_for_td:
            if prop != 'freq':
                self._compare(ser, prop)
        for prop in ok_for_td_methods:
            getattr(ser.dt, prop)
        result = ser.dt.components
        assert isinstance(result, DataFrame)
        tm.assert_index_equal(result.index, ser.index)
        result = ser.dt.to_pytimedelta()
        assert isinstance(result, np.ndarray)
        assert result.dtype == object
        result = ser.dt.total_seconds()
        assert isinstance(result, Series)
        assert result.dtype == 'float64'
        freq_result = ser.dt.freq
        assert freq_result == TimedeltaIndex(ser.values, freq='infer').freq