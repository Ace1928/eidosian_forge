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
def test_hour_index(self):
    dt_series = Series(date_range(start='2021-01-01', periods=5, freq='h'), index=[2, 6, 7, 8, 11], dtype='category')
    result = dt_series.dt.hour
    expected = Series([0, 1, 2, 3, 4], dtype='int32', index=[2, 6, 7, 8, 11])
    tm.assert_series_equal(result, expected)