from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_multi_timezone(self):
    df = DataFrame({'value': range(5), 'date': ['2000-01-28 16:47:00', '2000-01-29 16:48:00', '2000-01-30 16:49:00', '2000-01-31 16:50:00', '2000-01-01 16:50:00'], 'tz': ['America/Chicago', 'America/Chicago', 'America/Los_Angeles', 'America/Chicago', 'America/New_York']})
    result = df.groupby('tz', group_keys=False).date.apply(lambda x: pd.to_datetime(x).dt.tz_localize(x.name))
    expected = Series([Timestamp('2000-01-28 16:47:00-0600', tz='America/Chicago'), Timestamp('2000-01-29 16:48:00-0600', tz='America/Chicago'), Timestamp('2000-01-30 16:49:00-0800', tz='America/Los_Angeles'), Timestamp('2000-01-31 16:50:00-0600', tz='America/Chicago'), Timestamp('2000-01-01 16:50:00-0500', tz='America/New_York')], name='date', dtype=object)
    tm.assert_series_equal(result, expected)
    tz = 'America/Chicago'
    res_values = df.groupby('tz').date.get_group(tz)
    result = pd.to_datetime(res_values).dt.tz_localize(tz)
    exp_values = Series(['2000-01-28 16:47:00', '2000-01-29 16:48:00', '2000-01-31 16:50:00'], index=[0, 1, 3], name='date')
    expected = pd.to_datetime(exp_values).dt.tz_localize(tz)
    tm.assert_series_equal(result, expected)