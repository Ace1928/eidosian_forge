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
def test_evenly_divisible_with_no_extra_bins(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((9, 3)), index=date_range('2000-1-1', periods=9))
    result = df.resample('5D').mean()
    expected = pd.concat([df.iloc[0:5].mean(), df.iloc[5:].mean()], axis=1).T
    expected.index = pd.DatetimeIndex([Timestamp('2000-1-1'), Timestamp('2000-1-6')], dtype='M8[ns]', freq='5D')
    tm.assert_frame_equal(result, expected)