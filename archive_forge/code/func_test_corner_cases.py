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
def test_corner_cases(unit):
    rng = date_range('1/1/2000', periods=12, freq='min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('5min', closed='right', label='left').mean()
    ex_index = date_range('1999-12-31 23:55', periods=4, freq='5min').as_unit(unit)
    tm.assert_index_equal(result.index, ex_index)