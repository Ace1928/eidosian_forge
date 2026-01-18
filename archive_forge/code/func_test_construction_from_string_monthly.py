import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_construction_from_string_monthly(self):
    expected = date_range(start='2017-01-01', periods=5, freq='ME', name='foo').to_period()
    start, end = (str(expected[0]), str(expected[-1]))
    result = period_range(start=start, end=end, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(start=start, periods=5, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(end=end, periods=5, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    expected = PeriodIndex([], freq='M', name='foo')
    result = period_range(start=start, periods=0, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(end=end, periods=0, freq='M', name='foo')
    tm.assert_index_equal(result, expected)
    result = period_range(start=end, end=start, freq='M', name='foo')
    tm.assert_index_equal(result, expected)