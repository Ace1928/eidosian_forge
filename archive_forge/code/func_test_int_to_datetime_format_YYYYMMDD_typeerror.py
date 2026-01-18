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
@pytest.mark.parametrize('int_date, expected', [[20121030, datetime(2012, 10, 30)], [199934, datetime(1999, 3, 4)], [2012010101, 2012010101], [20129930, 20129930], [2012993, 2012993], [2121, 2121]])
def test_int_to_datetime_format_YYYYMMDD_typeerror(self, int_date, expected):
    result = to_datetime(int_date, format='%Y%m%d', errors='ignore')
    assert result == expected