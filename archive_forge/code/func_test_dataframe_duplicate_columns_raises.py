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
def test_dataframe_duplicate_columns_raises(self, cache):
    msg = 'cannot assemble with duplicate keys'
    df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
    df2.columns = ['year', 'year', 'day']
    with pytest.raises(ValueError, match=msg):
        to_datetime(df2, cache=cache)
    df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5], 'hour': [4, 5]})
    df2.columns = ['year', 'month', 'day', 'day']
    with pytest.raises(ValueError, match=msg):
        to_datetime(df2, cache=cache)