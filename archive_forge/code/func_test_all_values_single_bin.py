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
def test_all_values_single_bin(self):
    index = period_range(start='2012-01-01', end='2012-12-31', freq='M')
    ser = Series(np.random.default_rng(2).standard_normal(len(index)), index=index)
    result = ser.resample('Y').mean()
    tm.assert_almost_equal(result.iloc[0], ser.mean())