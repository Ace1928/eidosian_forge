from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_monotone_DTI_indexing_bug(self):
    df = DataFrame(list(range(5)))
    date_list = ['2018-01-02', '2017-02-10', '2016-03-10', '2015-03-15', '2014-03-16']
    date_index = DatetimeIndex(date_list)
    df['date'] = date_index
    expected = DataFrame({0: list(range(5)), 'date': date_index})
    tm.assert_frame_equal(df, expected)
    dti = date_range('20170101 01:00:00', periods=3)
    df = DataFrame({'A': [1, 2, 3]}, index=dti[::-1])
    expected = DataFrame({'A': 1}, index=dti[-1:][::-1])
    result = df.loc['2017-01-03']
    tm.assert_frame_equal(result, expected)
    result2 = df.iloc[::-1].loc['2017-01-03']
    expected2 = expected.iloc[::-1]
    tm.assert_frame_equal(result2, expected2)