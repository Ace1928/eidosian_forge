from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_closed_left_corner(self):
    s = Series(np.random.default_rng(2).standard_normal(21), index=date_range(start='1/1/2012 9:30', freq='1min', periods=21))
    s.iloc[0] = np.nan
    result = s.resample('10min', closed='left', label='right').mean()
    exp = s[1:].resample('10min', closed='left', label='right').mean()
    tm.assert_series_equal(result, exp)
    result = s.resample('10min', closed='left', label='left').mean()
    exp = s[1:].resample('10min', closed='left', label='left').mean()
    ex_index = date_range(start='1/1/2012 9:30', freq='10min', periods=3)
    tm.assert_index_equal(result.index, ex_index)
    tm.assert_series_equal(result, exp)