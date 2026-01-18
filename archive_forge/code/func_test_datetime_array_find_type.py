import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_array_find_type(self):
    dt = np.datetime64('1970-01-01', 'M')
    arr = np.array([dt])
    assert_equal(arr.dtype, np.dtype('M8[M]'))
    dt = datetime.date(1970, 1, 1)
    arr = np.array([dt])
    assert_equal(arr.dtype, np.dtype('O'))
    dt = datetime.datetime(1970, 1, 1, 12, 30, 40)
    arr = np.array([dt])
    assert_equal(arr.dtype, np.dtype('O'))
    b = np.bool_(True)
    dm = np.datetime64('1970-01-01', 'M')
    d = datetime.date(1970, 1, 1)
    dt = datetime.datetime(1970, 1, 1, 12, 30, 40)
    arr = np.array([b, dm])
    assert_equal(arr.dtype, np.dtype('O'))
    arr = np.array([b, d])
    assert_equal(arr.dtype, np.dtype('O'))
    arr = np.array([b, dt])
    assert_equal(arr.dtype, np.dtype('O'))
    arr = np.array([d, d]).astype('datetime64')
    assert_equal(arr.dtype, np.dtype('M8[D]'))
    arr = np.array([dt, dt]).astype('datetime64')
    assert_equal(arr.dtype, np.dtype('M8[us]'))