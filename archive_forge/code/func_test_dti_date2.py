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
def test_dti_date2(self, dtype):
    expected = np.array([date(2018, 6, 4), NaT])
    index = DatetimeIndex(['2018-06-04 10:00:00', NaT], dtype=dtype)
    result = index.date
    tm.assert_numpy_array_equal(result, expected)