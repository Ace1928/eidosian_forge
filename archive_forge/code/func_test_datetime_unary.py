import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_unary(self):
    for tda, tdb, tdzero, tdone, tdmone in [(np.array([3], dtype='m8[D]'), np.array([-3], dtype='m8[D]'), np.array([0], dtype='m8[D]'), np.array([1], dtype='m8[D]'), np.array([-1], dtype='m8[D]')), (np.timedelta64(3, '[D]'), np.timedelta64(-3, '[D]'), np.timedelta64(0, '[D]'), np.timedelta64(1, '[D]'), np.timedelta64(-1, '[D]'))]:
        assert_equal(-tdb, tda)
        assert_equal((-tdb).dtype, tda.dtype)
        assert_equal(np.negative(tdb), tda)
        assert_equal(np.negative(tdb).dtype, tda.dtype)
        assert_equal(np.positive(tda), tda)
        assert_equal(np.positive(tda).dtype, tda.dtype)
        assert_equal(np.positive(tdb), tdb)
        assert_equal(np.positive(tdb).dtype, tdb.dtype)
        assert_equal(np.absolute(tdb), tda)
        assert_equal(np.absolute(tdb).dtype, tda.dtype)
        assert_equal(np.sign(tda), tdone)
        assert_equal(np.sign(tdb), tdmone)
        assert_equal(np.sign(tdzero), tdzero)
        assert_equal(np.sign(tda).dtype, tda.dtype)
        assert_