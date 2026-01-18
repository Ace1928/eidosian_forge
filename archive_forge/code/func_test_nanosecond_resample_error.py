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
def test_nanosecond_resample_error():
    start = 1443707890427
    exp_start = 1443707890400
    indx = date_range(start=pd.to_datetime(start), periods=10, freq='100ns')
    ts = Series(range(len(indx)), index=indx)
    r = ts.resample(pd.tseries.offsets.Nano(100))
    result = r.agg('mean')
    exp_indx = date_range(start=pd.to_datetime(exp_start), periods=10, freq='100ns')
    exp = Series(range(len(exp_indx)), index=exp_indx, dtype=float)
    tm.assert_series_equal(result, exp)