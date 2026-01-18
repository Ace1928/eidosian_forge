import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_accepts_npcomplexfloating(self):
    assert_array_almost_equal(mgrid[0.1:0.3:3j,], mgrid[0.1:0.3:np.complex64(3j),])
    assert_array_almost_equal(mgrid[0.1:0.3:3j], mgrid[0.1:0.3:np.complex64(3j)])
    grid64_a = mgrid[0.1:0.3:3.3j]
    grid64_b = mgrid[0.1:0.3:3.3j,][0]
    assert_(grid64_a.dtype == grid64_b.dtype == np.float64)
    assert_array_equal(grid64_a, grid64_b)
    grid128_a = mgrid[0.1:0.3:np.clongdouble(3.3j)]
    grid128_b = mgrid[0.1:0.3:np.clongdouble(3.3j),][0]
    assert_(grid128_a.dtype == grid128_b.dtype == np.longdouble)
    assert_array_equal(grid64_a, grid64_b)