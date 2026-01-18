import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_accepts_npfloating(self):
    grid64 = mgrid[0.1:0.33:0.1,]
    grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1),]
    assert_(grid32.dtype == np.float64)
    assert_array_almost_equal(grid64, grid32)
    grid64 = mgrid[0.1:0.33:0.1]
    grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1)]
    assert_(grid32.dtype == np.float64)
    assert_array_almost_equal(grid64, grid32)