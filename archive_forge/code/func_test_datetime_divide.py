import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_divide(self):
    for dta, tda, tdb, tdc, tdd in [(np.array(['2012-12-21'], dtype='M8[D]'), np.array([6], dtype='m8[h]'), np.array([9], dtype='m8[h]'), np.array([12], dtype='m8[h]'), np.array([6], dtype='m8[m]')), (np.datetime64('2012-12-21', '[D]'), np.timedelta64(6, '[h]'), np.timedelta64(9, '[h]'), np.timedelta64(12, '[h]'), np.timedelta64(6, '[m]'))]:
        assert_equal(tdc / 2, tda)
        assert_equal((tdc / 2).dtype, np.dtype('m8[h]'))
        assert_equal(tda / 0.5, tdc)
        assert_equal((tda / 0.5).dtype, np.dtype('m8[h]'))
        assert_equal(tda / tdb, 6 / 9)
        assert_equal(np.divide(tda, tdb), 6 / 9)
        assert_equal(np.true_divide(tda, tdb), 6 / 9)
        assert_equal(tdb / tda, 9 / 6)
        assert_equal((tda / tdb).dtype, np.dtype('f8'))
        assert_equal(tda / tdd, 60)
        assert_equal(tdd / tda, 1 / 60)
        assert_raises(TypeError, np.divide, 2, tdb)
        assert_raises(TypeError, np.divide, 0.5, tdb)
        assert_raises(TypeError, np.divide, dta, tda)
        assert_raises(TypeError, np.divide, tda, dta)
        assert_raises(TypeError, np.divide, dta, 2)
        assert_raises(TypeError, np.divide, 2, dta)
        assert_raises(TypeError, np.divide, dta, 1.5)
        assert_raises(TypeError, np.divide, 1.5, dta)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, '.*encountered in divide')
        nat = np.timedelta64('NaT')
        for tp in (int, float):
            assert_equal(np.timedelta64(1) / tp(0), nat)
            assert_equal(np.timedelta64(0) / tp(0), nat)
            assert_equal(nat / tp(0), nat)
            assert_equal(nat / tp(2), nat)
        assert_equal(np.timedelta64(1) / float('inf'), np.timedelta64(0))
        assert_equal(np.timedelta64(0) / float('inf'), np.timedelta64(0))
        assert_equal(nat / float('inf'), nat)
        assert_equal(np.timedelta64(1) / float('nan'), nat)
        assert_equal(np.timedelta64(0) / float('nan'), nat)
        assert_equal(nat / float('nan'), nat)