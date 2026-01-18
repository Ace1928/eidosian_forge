import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_string_conversion(self):
    a = ['2011-03-16', '1920-01-01', '2013-05-19']
    str_a = np.array(a, dtype='S')
    uni_a = np.array(a, dtype='U')
    dt_a = np.array(a, dtype='M')
    assert_equal(dt_a, str_a.astype('M'))
    assert_equal(dt_a.dtype, str_a.astype('M').dtype)
    dt_b = np.empty_like(dt_a)
    dt_b[...] = str_a
    assert_equal(dt_a, dt_b)
    assert_equal(str_a, dt_a.astype('S0'))
    str_b = np.empty_like(str_a)
    str_b[...] = dt_a
    assert_equal(str_a, str_b)
    assert_equal(dt_a, uni_a.astype('M'))
    assert_equal(dt_a.dtype, uni_a.astype('M').dtype)
    dt_b = np.empty_like(dt_a)
    dt_b[...] = uni_a
    assert_equal(dt_a, dt_b)
    assert_equal(uni_a, dt_a.astype('U'))
    uni_b = np.empty_like(uni_a)
    uni_b[...] = dt_a
    assert_equal(uni_a, uni_b)
    assert_equal(str_a, dt_a.astype((np.bytes_, 128)))
    str_b = np.empty(str_a.shape, dtype=(np.bytes_, 128))
    str_b[...] = dt_a
    assert_equal(str_a, str_b)