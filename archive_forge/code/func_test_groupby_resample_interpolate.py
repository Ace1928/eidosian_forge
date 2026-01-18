from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
def test_groupby_resample_interpolate():
    d = {'price': [10, 11, 9], 'volume': [50, 60, 50]}
    df = DataFrame(d)
    df['week_starting'] = date_range('01/01/2018', periods=3, freq='W')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.set_index('week_starting').groupby('volume').resample('1D').interpolate(method='linear')
    volume = [50] * 15 + [60]
    week_starting = list(date_range('2018-01-07', '2018-01-21')) + [Timestamp('2018-01-14')]
    expected_ind = pd.MultiIndex.from_arrays([volume, week_starting], names=['volume', 'week_starting'])
    expected = DataFrame(data={'price': [10.0, 9.928571428571429, 9.857142857142858, 9.785714285714286, 9.714285714285714, 9.642857142857142, 9.571428571428571, 9.5, 9.428571428571429, 9.357142857142858, 9.285714285714286, 9.214285714285714, 9.142857142857142, 9.071428571428571, 9.0, 11.0], 'volume': [50.0] * 15 + [60]}, index=expected_ind)
    tm.assert_frame_equal(result, expected)