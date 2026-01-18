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
@pytest.mark.slow
@pytest.mark.parametrize('s_ts, e_ts', [('2262-02-23', '1969-11-14'), ('1970-02-01', '1677-10-22')])
def test_date_range_int64_overflow_stride_endpoint_different_signs(self, s_ts, e_ts):
    start = Timestamp(s_ts)
    end = Timestamp(e_ts)
    expected = date_range(start=start, end=end, freq='-1h')
    assert expected[0] == start
    assert expected[-1] == end
    dti = date_range(end=end, periods=len(expected), freq='-1h')
    tm.assert_index_equal(dti, expected)