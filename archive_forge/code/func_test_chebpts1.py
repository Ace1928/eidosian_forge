from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebpts1(self):
    assert_raises(ValueError, cheb.chebpts1, 1.5)
    assert_raises(ValueError, cheb.chebpts1, 0)
    tgt = [0]
    assert_almost_equal(cheb.chebpts1(1), tgt)
    tgt = [-0.7071067811865475, 0.7071067811865475]
    assert_almost_equal(cheb.chebpts1(2), tgt)
    tgt = [-0.8660254037844387, 0, 0.8660254037844387]
    assert_almost_equal(cheb.chebpts1(3), tgt)
    tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
    assert_almost_equal(cheb.chebpts1(4), tgt)