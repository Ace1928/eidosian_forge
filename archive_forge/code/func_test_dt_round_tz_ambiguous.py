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
@pytest.mark.parametrize('method', ['ceil', 'round', 'floor'])
def test_dt_round_tz_ambiguous(self, method):
    df1 = DataFrame([pd.to_datetime('2017-10-29 02:00:00+02:00', utc=True), pd.to_datetime('2017-10-29 02:00:00+01:00', utc=True), pd.to_datetime('2017-10-29 03:00:00+01:00', utc=True)], columns=['date'])
    df1['date'] = df1['date'].dt.tz_convert('Europe/Madrid')
    result = getattr(df1.date.dt, method)('h', ambiguous='infer')
    expected = df1['date']
    tm.assert_series_equal(result, expected)
    result = getattr(df1.date.dt, method)('h', ambiguous=[True, False, False])
    tm.assert_series_equal(result, expected)
    result = getattr(df1.date.dt, method)('h', ambiguous='NaT')
    expected = df1['date'].copy()
    expected.iloc[0:2] = pd.NaT
    tm.assert_series_equal(result, expected)
    with tm.external_error_raised(pytz.AmbiguousTimeError):
        getattr(df1.date.dt, method)('h', ambiguous='raise')