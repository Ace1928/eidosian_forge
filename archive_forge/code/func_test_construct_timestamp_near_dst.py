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
@pytest.mark.parametrize('offset', ['+0300', '+0200'])
def test_construct_timestamp_near_dst(self, offset):
    expected = Timestamp(f'2016-10-30 03:00:00{offset}', tz='Europe/Helsinki')
    result = Timestamp(expected).tz_convert('Europe/Helsinki')
    assert result == expected