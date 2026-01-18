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
@pytest.mark.parametrize('dtype', [None, 'datetime64[ns, CET]', 'datetime64[ns, EST]', 'datetime64[ns, UTC]'])
def test_dti_time2(self, dtype):
    expected = np.array([time(10, 20, 30), NaT])
    index = DatetimeIndex(['2018-06-04 10:20:30', NaT], dtype=dtype)
    result = index.time
    tm.assert_numpy_array_equal(result, expected)