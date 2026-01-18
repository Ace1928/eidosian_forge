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
def test_dayfirst(self, cache):
    arr = ['10/02/2014', '11/02/2014', '12/02/2014']
    expected = DatetimeIndex([datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)])
    idx1 = DatetimeIndex(arr, dayfirst=True)
    idx2 = DatetimeIndex(np.array(arr), dayfirst=True)
    idx3 = to_datetime(arr, dayfirst=True, cache=cache)
    idx4 = to_datetime(np.array(arr), dayfirst=True, cache=cache)
    idx5 = DatetimeIndex(Index(arr), dayfirst=True)
    idx6 = DatetimeIndex(Series(arr), dayfirst=True)
    tm.assert_index_equal(expected, idx1)
    tm.assert_index_equal(expected, idx2)
    tm.assert_index_equal(expected, idx3)
    tm.assert_index_equal(expected, idx4)
    tm.assert_index_equal(expected, idx5)
    tm.assert_index_equal(expected, idx6)