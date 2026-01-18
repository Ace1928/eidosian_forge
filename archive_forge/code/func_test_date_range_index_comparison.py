from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_date_range_index_comparison(self):
    rng = date_range('2011-01-01', periods=3, tz='US/Eastern')
    df = Series(rng).to_frame()
    arr = np.array([rng.to_list()]).T
    arr2 = np.array([rng]).T
    with pytest.raises(ValueError, match='Unable to coerce to Series'):
        rng == df
    with pytest.raises(ValueError, match='Unable to coerce to Series'):
        df == rng
    expected = DataFrame([True, True, True])
    results = df == arr2
    tm.assert_frame_equal(results, expected)
    expected = Series([True, True, True], name=0)
    results = df[0] == arr2[:, 0]
    tm.assert_series_equal(results, expected)
    expected = np.array([[True, False, False], [False, True, False], [False, False, True]])
    results = rng == arr
    tm.assert_numpy_array_equal(results, expected)