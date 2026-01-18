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
def test_downsample_across_dst(unit):
    tz = pytz.timezone('Europe/Berlin')
    dt = datetime(2014, 10, 26)
    dates = date_range(tz.localize(dt), periods=4, freq='2h').as_unit(unit)
    result = Series(5, index=dates).resample('h').mean()
    expected = Series([5.0, np.nan] * 3 + [5.0], index=date_range(tz.localize(dt), periods=7, freq='h').as_unit(unit))
    tm.assert_series_equal(result, expected)