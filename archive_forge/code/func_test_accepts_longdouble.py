import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_accepts_longdouble(self):
    grid64 = mgrid[0.1:0.33:0.1,]
    grid128 = mgrid[np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1),]
    assert_(grid128.dtype == np.longdouble)
    assert_array_almost_equal(grid64, grid128)
    grid128c_a = mgrid[0:np.longdouble(1):3.4j]
    grid128c_b = mgrid[0:np.longdouble(1):3.4j,]
    assert_(grid128c_a.dtype == grid128c_b.dtype == np.longdouble)
    assert_array_equal(grid128c_a, grid128c_b[0])
    grid64 = mgrid[0.1:0.33:0.1]
    grid128 = mgrid[np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1)]
    assert_(grid128.dtype == np.longdouble)
    assert_array_almost_equal(grid64, grid128)