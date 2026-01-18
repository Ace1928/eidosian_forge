import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq_offset, freq_period', [('D', 'D'), ('W', 'W'), ('QE', 'Q'), ('YE', 'Y')])
def test_construction_from_string(self, freq_offset, freq_period):
    expected = date_range(start='2017-01-01', periods=5, freq=freq_offset, name='foo').to_period()
    start, end = (str(expected[0]), str(expected[-1]))
    result = period_range(start=start, end=end, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(start=start, periods=5, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(end=end, periods=5, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)
    expected = PeriodIndex([], freq=freq_period, name='foo')
    result = period_range(start=start, periods=0, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(end=end, periods=0, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(start=end, end=start, freq=freq_period, name='foo')
    tm.assert_index_equal(result, expected)