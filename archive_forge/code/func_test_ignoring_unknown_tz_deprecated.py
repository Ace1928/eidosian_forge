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
def test_ignoring_unknown_tz_deprecated():
    dtstr = '2014 Jan 9 05:15 FAKE'
    msg = 'un-recognized timezone "FAKE". Dropping unrecognized timezones is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = Timestamp(dtstr)
    assert res == Timestamp(dtstr[:-5])
    with tm.assert_produces_warning(FutureWarning):
        res = to_datetime(dtstr)
    assert res == to_datetime(dtstr[:-5])
    with tm.assert_produces_warning(FutureWarning):
        res = to_datetime([dtstr])
    tm.assert_index_equal(res, to_datetime([dtstr[:-5]]))