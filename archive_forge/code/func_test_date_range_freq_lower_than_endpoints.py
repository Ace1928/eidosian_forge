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
def test_date_range_freq_lower_than_endpoints(self):
    start = Timestamp('2022-10-19 11:50:44.719781')
    end = Timestamp('2022-10-19 11:50:47.066458')
    with pytest.raises(ValueError, match='Cannot losslessly convert units'):
        date_range(start, end, periods=3, unit='s')
    dti = date_range(start, end, periods=2, unit='us')
    rng = np.array([start.as_unit('us')._value, end.as_unit('us')._value], dtype=np.int64)
    expected = DatetimeIndex(rng.view('M8[us]'))
    tm.assert_index_equal(dti, expected)