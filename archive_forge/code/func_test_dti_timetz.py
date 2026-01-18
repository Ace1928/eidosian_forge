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
def test_dti_timetz(self, tz_naive_fixture):
    tz = timezones.maybe_get_tz(tz_naive_fixture)
    expected = np.array([time(10, 20, 30, tzinfo=tz), NaT])
    index = DatetimeIndex(['2018-06-04 10:20:30', NaT], tz=tz)
    result = index.timetz
    tm.assert_numpy_array_equal(result, expected)