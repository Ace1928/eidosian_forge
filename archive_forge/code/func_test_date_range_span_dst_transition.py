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
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_date_range_span_dst_transition(self, tzstr):
    dr = date_range('03/06/2012 00:00', periods=200, freq='W-FRI', tz='US/Eastern')
    assert (dr.hour == 0).all()
    dr = date_range('2012-11-02', periods=10, tz=tzstr)
    result = dr.hour
    expected = pd.Index([0] * 10, dtype='int32')
    tm.assert_index_equal(result, expected)