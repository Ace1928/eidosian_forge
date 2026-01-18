import pytest
from pandas import (
@pytest.mark.parametrize('left, right', [('a', 'z'), (('a', 'b'), ('c', 'd')), (list('AB'), list('ab')), (Interval(0, 1), Interval(1, 2)), (Period('2018Q1', freq='Q'), Period('2018Q1', freq='Q'))])
def test_construct_errors(self, left, right):
    msg = 'Only numeric, Timestamp and Timedelta endpoints are allowed'
    with pytest.raises(ValueError, match=msg):
        Interval(left, right)