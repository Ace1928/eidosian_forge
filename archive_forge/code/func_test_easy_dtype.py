import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_easy_dtype(self):
    """Test ndtype on dtypes"""
    ndtype = float
    assert_equal(easy_dtype(ndtype), np.dtype(float))
    ndtype = 'i4, f8'
    assert_equal(easy_dtype(ndtype), np.dtype([('f0', 'i4'), ('f1', 'f8')]))
    assert_equal(easy_dtype(ndtype, defaultfmt='field_%03i'), np.dtype([('field_000', 'i4'), ('field_001', 'f8')]))
    ndtype = 'i4, f8'
    assert_equal(easy_dtype(ndtype, names='a, b'), np.dtype([('a', 'i4'), ('b', 'f8')]))
    ndtype = 'i4, f8'
    assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', 'i4'), ('b', 'f8')]))
    ndtype = 'i4, f8'
    assert_equal(easy_dtype(ndtype, names=', b'), np.dtype([('f0', 'i4'), ('b', 'f8')]))
    assert_equal(easy_dtype(ndtype, names='a', defaultfmt='f%02i'), np.dtype([('a', 'i4'), ('f00', 'f8')]))
    ndtype = [('A', int), ('B', float)]
    assert_equal(easy_dtype(ndtype), np.dtype([('A', int), ('B', float)]))
    assert_equal(easy_dtype(ndtype, names='a,b'), np.dtype([('a', int), ('b', float)]))
    assert_equal(easy_dtype(ndtype, names='a'), np.dtype([('a', int), ('f0', float)]))
    assert_equal(easy_dtype(ndtype, names='a,b,c'), np.dtype([('a', int), ('b', float)]))
    ndtype = (int, float, float)
    assert_equal(easy_dtype(ndtype), np.dtype([('f0', int), ('f1', float), ('f2', float)]))
    ndtype = (int, float, float)
    assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', int), ('b', float), ('c', float)]))
    ndtype = np.dtype(float)
    assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([(_, float) for _ in ('a', 'b', 'c')]))
    ndtype = np.dtype(float)
    assert_equal(easy_dtype(ndtype, names=['', '', ''], defaultfmt='f%02i'), np.dtype([(_, float) for _ in ('f00', 'f01', 'f02')]))