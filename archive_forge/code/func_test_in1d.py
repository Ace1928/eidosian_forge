import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('kind', [None, 'sort', 'table'])
def test_in1d(self, kind):
    for mult in (1, 10):
        a = [5, 7, 1, 2]
        b = [2, 4, 3, 1, 5] * mult
        ec = np.array([True, False, True, True])
        c = in1d(a, b, assume_unique=True, kind=kind)
        assert_array_equal(c, ec)
        a[0] = 8
        ec = np.array([False, False, True, True])
        c = in1d(a, b, assume_unique=True, kind=kind)
        assert_array_equal(c, ec)
        a[0], a[3] = (4, 8)
        ec = np.array([True, False, True, False])
        c = in1d(a, b, assume_unique=True, kind=kind)
        assert_array_equal(c, ec)
        a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
        b = [2, 3, 4] * mult
        ec = [False, True, False, True, True, True, True, True, True, False, True, False, False, False]
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
        b = b + [5, 5, 4] * mult
        ec = [True, True, True, True, True, True, True, True, True, True, True, False, True, True]
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5] * mult)
        ec = np.array([True, False, True, True])
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
        a = np.array([5, 7, 1, 1, 2])
        b = np.array([2, 4, 3, 3, 1, 5] * mult)
        ec = np.array([True, False, True, True, True])
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
        a = np.array([5, 5])
        b = np.array([2, 2] * mult)
        ec = np.array([False, False])
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
    a = np.array([5])
    b = np.array([2])
    ec = np.array([False])
    c = in1d(a, b, kind=kind)
    assert_array_equal(c, ec)
    if kind in {None, 'sort'}:
        assert_array_equal(in1d([], [], kind=kind), [])