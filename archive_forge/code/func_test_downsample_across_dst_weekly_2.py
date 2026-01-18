from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_downsample_across_dst_weekly_2(unit):
    idx = date_range('2013-04-01', '2013-05-01', tz='Europe/London', freq='h').as_unit(unit)
    s = Series(index=idx, dtype=np.float64)
    result = s.resample('W').mean()
    expected = Series(index=date_range('2013-04-07', freq='W', periods=5, tz='Europe/London').as_unit(unit), dtype=np.float64)
    tm.assert_series_equal(result, expected)