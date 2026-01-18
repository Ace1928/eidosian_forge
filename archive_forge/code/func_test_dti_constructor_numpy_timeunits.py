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
@pytest.mark.parametrize('unit', ['h', 'm', 's', 'ms', 'us', 'ns'])
def test_dti_constructor_numpy_timeunits(self, cache, unit):
    dtype = np.dtype(f'M8[{unit}]')
    base = to_datetime(['2000-01-01T00:00', '2000-01-02T00:00', 'NaT'], cache=cache)
    values = base.values.astype(dtype)
    if unit in ['h', 'm']:
        unit = 's'
    exp_dtype = np.dtype(f'M8[{unit}]')
    expected = DatetimeIndex(base.astype(exp_dtype))
    assert expected.dtype == exp_dtype
    tm.assert_index_equal(DatetimeIndex(values), expected)
    tm.assert_index_equal(to_datetime(values, cache=cache), expected)