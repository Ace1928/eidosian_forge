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
@pytest.mark.parametrize('freq', ['ns', 'us', 'ms', 'min', 's', 'h', 'D'])
def test_date_range_edges(self, freq):
    td = Timedelta(f'1{freq}')
    ts = Timestamp('1970-01-01')
    idx = date_range(start=ts + td, end=ts + 4 * td, freq=freq)
    exp = DatetimeIndex([ts + n * td for n in range(1, 5)], dtype='M8[ns]', freq=freq)
    tm.assert_index_equal(idx, exp)
    idx = date_range(start=ts + 4 * td, end=ts + td, freq=freq)
    exp = DatetimeIndex([], dtype='M8[ns]', freq=freq)
    tm.assert_index_equal(idx, exp)
    idx = date_range(start=ts + td, end=ts + td, freq=freq)
    exp = DatetimeIndex([ts + td], dtype='M8[ns]', freq=freq)
    tm.assert_index_equal(idx, exp)