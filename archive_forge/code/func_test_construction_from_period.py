import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_construction_from_period(self):
    start, end = (Period('2017Q1', freq='Q'), Period('2018Q1', freq='Q'))
    expected = date_range(start='2017-03-31', end='2018-03-31', freq='ME', name='foo').to_period()
    result = period_range(start=start, end=end, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    start = Period('2017-1', freq='M')
    end = Period('2019-12', freq='M')
    expected = date_range(start='2017-01-31', end='2019-12-31', freq='QE', name='foo').to_period()
    result = period_range(start=start, end=end, freq='Q', name='foo')
    tm.assert_index_equal(result, expected)
    start = Period('2017Q1', freq='Q')
    end = Period('2018Q1', freq='Q')
    idx = period_range(start=start, end=end, freq='Q', name='foo')
    result = idx == idx.values
    expected = np.array([True, True, True, True, True])
    tm.assert_numpy_array_equal(result, expected)
    expected = PeriodIndex([], freq='W', name='foo')
    result = period_range(start=start, periods=0, freq='W', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(end=end, periods=0, freq='W', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(start=end, end=start, freq='W', name='foo')
    tm.assert_index_equal(result, expected)