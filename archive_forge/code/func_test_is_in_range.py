import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.link import (
@pytest.mark.parametrize('interval', [Interval(0, 1, False, False), Interval(0, 1, False, True), Interval(0, 1, True, False), Interval(0, 1, True, True), Interval(-np.inf, np.inf, False, False), Interval(-np.inf, np.inf, False, True), Interval(-np.inf, np.inf, True, False), Interval(-np.inf, np.inf, True, True), Interval(-10, -1, False, False), Interval(-10, -1, False, True), Interval(-10, -1, True, False), Interval(-10, -1, True, True)])
def test_is_in_range(interval):
    low, high = _inclusive_low_high(interval)
    x = np.linspace(low, high, num=10)
    assert interval.includes(x)
    assert interval.includes(np.r_[x, interval.low]) == interval.low_inclusive
    assert interval.includes(np.r_[x, interval.high]) == interval.high_inclusive
    assert interval.includes(np.r_[x, interval.low, interval.high]) == (interval.low_inclusive and interval.high_inclusive)