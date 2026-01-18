from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
@pytest.mark.parametrize('n,date', [(2, '1862-01-13 09:03:34.873477378+0210'), (-2, '1856-10-24 16:18:36.556360110-0717')])
def test_is_on_offset_weekday_none(self, n, date):
    offset = Week(n=n, weekday=None)
    ts = Timestamp(date, tz='Africa/Lusaka')
    fast = offset.is_on_offset(ts)
    slow = ts + offset - offset == ts
    assert fast == slow