import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [[Timestamp('2013-01-01'), Timestamp('2013-01-02'), Timestamp('2013-01-03')], [Timedelta('5 days'), Timedelta('6 days'), Timedelta('7 days')]])
def test_group_diff_datetimelike(data):
    df = DataFrame({'a': [1, 2, 2], 'b': data})
    result = df.groupby('a')['b'].diff()
    expected = Series([NaT, NaT, Timedelta('1 days')], name='b')
    tm.assert_series_equal(result, expected)