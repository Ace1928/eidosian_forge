from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
@pytest.mark.parametrize('n,week,date,tz', [(2, 2, '1916-05-15 01:14:49.583410462+0422', 'Asia/Qyzylorda'), (-3, 1, '1980-12-08 03:38:52.878321185+0500', 'Asia/Oral')])
def test_is_on_offset_nanoseconds(self, n, week, date, tz):
    offset = WeekOfMonth(n=n, week=week, weekday=0)
    ts = Timestamp(date, tz=tz)
    fast = offset.is_on_offset(ts)
    slow = ts + offset - offset == ts
    assert fast == slow