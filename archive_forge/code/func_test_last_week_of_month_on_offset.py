from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
@pytest.mark.parametrize('n,weekday,date,tz', [(4, 6, '1917-05-27 20:55:27.084284178+0200', 'Europe/Warsaw'), (-4, 5, '2005-08-27 05:01:42.799392561-0500', 'America/Rainy_River')])
def test_last_week_of_month_on_offset(self, n, weekday, date, tz):
    offset = LastWeekOfMonth(n=n, weekday=weekday)
    ts = Timestamp(date, tz=tz)
    slow = ts + offset - offset == ts
    fast = offset.is_on_offset(ts)
    assert fast == slow