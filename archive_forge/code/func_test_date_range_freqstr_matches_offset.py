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
@pytest.mark.parametrize('freqstr,offset', [('QS', offsets.QuarterBegin(startingMonth=1)), ('BQE', offsets.BQuarterEnd(startingMonth=12)), ('W-SUN', offsets.Week(weekday=6))])
def test_date_range_freqstr_matches_offset(self, freqstr, offset):
    sdate = datetime(1999, 12, 25)
    edate = datetime(2000, 1, 1)
    idx1 = date_range(start=sdate, end=edate, freq=freqstr)
    idx2 = date_range(start=sdate, end=edate, freq=offset)
    assert len(idx1) == len(idx2)
    assert idx1.freq == idx2.freq