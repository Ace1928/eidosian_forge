import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
def test_mixed_offsets_with_native_datetime_raises(self):
    vals = ['nan', Timestamp('1990-01-01'), '2015-03-14T16:15:14.123-08:00', '2019-03-04T21:56:32.620-07:00', None, 'today', 'now']
    ser = Series(vals)
    assert all((ser[i] is vals[i] for i in range(len(vals))))
    now = Timestamp('now')
    today = Timestamp('today')
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        mixed = to_datetime(ser)
    expected = Series(['NaT', Timestamp('1990-01-01'), Timestamp('2015-03-14T16:15:14.123-08:00').to_pydatetime(), Timestamp('2019-03-04T21:56:32.620-07:00').to_pydatetime(), None], dtype=object)
    tm.assert_series_equal(mixed[:-2], expected)
    assert (now - mixed.iloc[-1]).total_seconds() <= 0.1
    assert (today - mixed.iloc[-2]).total_seconds() <= 0.1
    with pytest.raises(ValueError, match='Tz-aware datetime.datetime'):
        to_datetime(mixed)