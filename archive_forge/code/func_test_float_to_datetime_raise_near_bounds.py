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
def test_float_to_datetime_raise_near_bounds(self):
    msg = "cannot convert input with unit 'D'"
    oneday_in_ns = 1000000000.0 * 60 * 60 * 24
    tsmax_in_days = 2 ** 63 / oneday_in_ns
    should_succeed = Series([0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float)
    expected = (should_succeed * oneday_in_ns).astype(np.int64)
    for error_mode in ['raise', 'coerce', 'ignore']:
        result1 = to_datetime(should_succeed, unit='D', errors=error_mode)
        tm.assert_almost_equal(result1.astype(np.int64).astype(np.float64), expected.astype(np.float64), rtol=1e-10)
    should_fail1 = Series([0, tsmax_in_days + 0.005], dtype=float)
    should_fail2 = Series([0, -tsmax_in_days - 0.005], dtype=float)
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime(should_fail1, unit='D', errors='raise')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime(should_fail2, unit='D', errors='raise')