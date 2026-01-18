import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('epoch', [1552211999999999872, 1552211999999999999])
def test_constructor_before_dst_switch(epoch):
    ts = Timestamp(epoch, tz='dateutil/America/Los_Angeles')
    result = ts.tz.dst(ts)
    expected = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch
    assert result == expected