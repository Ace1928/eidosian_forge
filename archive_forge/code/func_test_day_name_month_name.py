import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('time_locale', [None] + tm.get_locales())
def test_day_name_month_name(self, time_locale):
    if time_locale is None:
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        expected_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    else:
        with tm.set_locale(time_locale, locale.LC_TIME):
            expected_days = calendar.day_name[:]
            expected_months = calendar.month_name[1:]
    dti = date_range(freq='D', start=datetime(1998, 1, 1), periods=365)
    english_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
        name = name.capitalize()
        assert dti.day_name(locale=time_locale)[day] == name
        assert dti.day_name(locale=None)[day] == eng_name
        ts = Timestamp(datetime(2016, 4, day))
        assert ts.day_name(locale=time_locale) == name
    dti = dti.append(DatetimeIndex([NaT]))
    assert np.isnan(dti.day_name(locale=time_locale)[-1])
    ts = Timestamp(NaT)
    assert np.isnan(ts.day_name(locale=time_locale))
    dti = date_range(freq='ME', start='2012', end='2013')
    result = dti.month_name(locale=time_locale)
    expected = Index([month.capitalize() for month in expected_months])
    result = result.str.normalize('NFD')
    expected = expected.str.normalize('NFD')
    tm.assert_index_equal(result, expected)
    for item, expected in zip(dti, expected_months):
        result = item.month_name(locale=time_locale)
        expected = expected.capitalize()
        result = unicodedata.normalize('NFD', result)
        expected = unicodedata.normalize('NFD', result)
        assert result == expected
    dti = dti.append(DatetimeIndex([NaT]))
    assert np.isnan(dti.month_name(locale=time_locale)[-1])