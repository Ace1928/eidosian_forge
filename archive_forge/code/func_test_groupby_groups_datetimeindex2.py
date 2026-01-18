from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_groups_datetimeindex2(self):
    index = date_range('2015/01/01', periods=5, name='date')
    df = DataFrame({'A': [5, 6, 7, 8, 9], 'B': [1, 2, 3, 4, 5]}, index=index)
    result = df.groupby(level='date').groups
    dates = ['2015-01-05', '2015-01-04', '2015-01-03', '2015-01-02', '2015-01-01']
    expected = {Timestamp(date): DatetimeIndex([date], name='date') for date in dates}
    tm.assert_dict_equal(result, expected)
    grouped = df.groupby(level='date')
    for date in dates:
        result = grouped.get_group(date)
        data = [[df.loc[date, 'A'], df.loc[date, 'B']]]
        expected_index = DatetimeIndex([date], name='date', freq='D', dtype=index.dtype)
        expected = DataFrame(data, columns=list('AB'), index=expected_index)
        tm.assert_frame_equal(result, expected)