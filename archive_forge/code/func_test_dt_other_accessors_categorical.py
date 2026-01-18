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
@pytest.mark.parametrize('accessor', ['year', 'month', 'day'])
def test_dt_other_accessors_categorical(self, accessor):
    datetimes = Series(['2018-01-01', '2018-01-01', '2019-01-02'], dtype='datetime64[ns]')
    categorical = datetimes.astype('category')
    result = getattr(categorical.dt, accessor)
    expected = getattr(datetimes.dt, accessor)
    tm.assert_series_equal(result, expected)