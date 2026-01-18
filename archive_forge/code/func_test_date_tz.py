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
def test_date_tz(self):
    rng = DatetimeIndex(['2014-04-04 23:56', '2014-07-18 21:24', '2015-11-22 22:14'], tz='US/Eastern')
    ser = Series(rng)
    expected = Series([date(2014, 4, 4), date(2014, 7, 18), date(2015, 11, 22)])
    tm.assert_series_equal(ser.dt.date, expected)
    tm.assert_series_equal(ser.apply(lambda x: x.date()), expected)