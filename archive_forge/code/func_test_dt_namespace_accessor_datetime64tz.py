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
def test_dt_namespace_accessor_datetime64tz(self):
    dti = date_range('20130101', periods=5, tz='US/Eastern')
    ser = Series(dti, name='xxx')
    for prop in ok_for_dt:
        if prop != 'freq':
            self._compare(ser, prop)
    for prop in ok_for_dt_methods:
        getattr(ser.dt, prop)
    msg = 'The behavior of DatetimeProperties.to_pydatetime is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.dt.to_pydatetime()
    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    result = ser.dt.tz_convert('CET')
    expected = Series(ser._values.tz_convert('CET'), index=ser.index, name='xxx')
    tm.assert_series_equal(result, expected)
    tz_result = result.dt.tz
    assert str(tz_result) == 'CET'
    freq_result = ser.dt.freq
    assert freq_result == DatetimeIndex(ser.values, freq='infer').freq