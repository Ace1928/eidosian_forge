import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_add(self):
    for dta, dtb, dtc, dtnat, tda, tdb, tdc in [(np.array(['2012-12-21'], dtype='M8[D]'), np.array(['2012-12-24'], dtype='M8[D]'), np.array(['2012-12-21T11'], dtype='M8[h]'), np.array(['NaT'], dtype='M8[D]'), np.array([3], dtype='m8[D]'), np.array([11], dtype='m8[h]'), np.array([3 * 24 + 11], dtype='m8[h]')), (np.datetime64('2012-12-21', '[D]'), np.datetime64('2012-12-24', '[D]'), np.datetime64('2012-12-21T11', '[h]'), np.datetime64('NaT', '[D]'), np.timedelta64(3, '[D]'), np.timedelta64(11, '[h]'), np.timedelta64(3 * 24 + 11, '[h]'))]:
        assert_equal(tda + tdb, tdc)
        assert_equal((tda + tdb).dtype, np.dtype('m8[h]'))
        assert_equal(tdb + True, tdb + 1)
        assert_equal((tdb + True).dtype, np.dtype('m8[h]'))
        assert_equal(tdb + 3 * 24, tdc)
        assert_equal((tdb + 3 * 24).dtype, np.dtype('m8[h]'))
        assert_equal(False + tdb, tdb)
        assert_equal((False + tdb).dtype, np.dtype('m8[h]'))
        assert_equal(3 * 24 + tdb, tdc)
        assert_equal((3 * 24 + tdb).dtype, np.dtype('m8[h]'))
        assert_equal(dta + True, dta + 1)
        assert_equal(dtnat + True, dtnat)
        assert_equal((dta + True).dtype, np.dtype('M8[D]'))
        assert_equal(dta + 3, dtb)
        assert_equal(dtnat + 3, dtnat)
        assert_equal((dta + 3).dtype, np.dtype('M8[D]'))
        assert_equal(False + dta, dta)
        assert_equal(False + dtnat, dtnat)
        assert_equal((False + dta).dtype, np.dtype('M8[D]'))
        assert_equal(3 + dta, dtb)
        assert_equal(3 + dtnat, dtnat)
        assert_equal((3 + dta).dtype, np.dtype('M8[D]'))
        assert_equal(dta + tda, dtb)
        assert_equal(dtnat + tda, dtnat)
        assert_equal((dta + tda).dtype, np.dtype('M8[D]'))
        assert_equal(tda + dta, dtb)
        assert_equal(tda + dtnat, dtnat)
        assert_equal((tda + dta).dtype, np.dtype('M8[D]'))
        assert_equal(np.add(dta, tdb, casting='unsafe'), dtc)
        assert_equal(np.add(dta, tdb, casting='unsafe').dtype, np.dtype('M8[h]'))
        assert_equal(np.add(tdb, dta, casting='unsafe'), dtc)
        assert_equal(np.add(tdb, dta, casting='unsafe').dtype, np.dtype('M8[h]'))
        assert_raises(TypeError, np.add, dta, dtb)