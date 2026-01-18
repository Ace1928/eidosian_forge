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
def test_dt_accessor_updates_on_inplace(self):
    ser = Series(date_range('2018-01-01', periods=10))
    ser[2] = None
    return_value = ser.fillna(pd.Timestamp('2018-01-01'), inplace=True)
    assert return_value is None
    result = ser.dt.date
    assert result[0] == result[2]