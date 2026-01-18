import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_multiply(self):
    for dta, tda, tdb, tdc in [(np.array(['2012-12-21'], dtype='M8[D]'), np.array([6], dtype='m8[h]'), np.array([9], dtype='m8[h]'), np.array([12], dtype='m8[h]')), (np.datetime64('2012-12-21', '[D]'), np.timedelta64(6, '[h]'), np.timedelta64(9, '[h]'), np.timedelta64(12, '[h]'))]:
        assert_equal(tda * 2, tdc)
        assert_equal((tda * 2).dtype, np.dtype('m8[h]'))
        assert_equal(2 * tda, tdc)
        assert_equal((2 * tda).dtype, np.dtype('m8[h]'))
        assert_equal(tda * 1.5, tdb)
        assert_equal((tda * 1.5).dtype, np.dtype('m8[h]'))
        assert_equal(1.5 * tda, tdb)
        assert_equal((1.5 * tda).dtype, np.dtype('m8[h]'))
        assert_raises(TypeError, np.multiply, tda, tdb)
        assert_raises(TypeError, np.multiply, dta, tda)
        assert_raises(TypeError, np.multiply, tda, dta)
        assert_raises(TypeError, np.multiply, dta, 2)
        assert_raises(TypeError, np.multiply, 2, dta)
        assert_raises(TypeError, np.multiply, dta, 1.5)
        assert_raises(TypeError, np.multiply, 1.5, dta)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
        nat = np.timedelta64('NaT')

        def check(a, b, res):
            assert_equal(a * b, res)
            assert_equal(b * a, res)
        for tp in (int, float):
            check(nat, tp(2), nat)
            check(nat, tp(0), nat)
        for f in (float('inf'), float('nan')):
            check(np.timedelta64(1), f, nat)
            check(np.timedelta64(0), f, nat)
            check(nat, f, nat)