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
def test_downsample_across_dst_weekly(unit):
    df = DataFrame(index=DatetimeIndex(['2017-03-25', '2017-03-26', '2017-03-27', '2017-03-28', '2017-03-29'], tz='Europe/Amsterdam').as_unit(unit), data=[11, 12, 13, 14, 15])
    result = df.resample('1W').sum()
    expected = DataFrame([23, 42], index=DatetimeIndex(['2017-03-26', '2017-04-02'], tz='Europe/Amsterdam', freq='W').as_unit(unit))
    tm.assert_frame_equal(result, expected)