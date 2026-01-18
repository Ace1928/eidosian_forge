from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_cheb2poly(self):
    for i in range(10):
        assert_almost_equal(cheb.cheb2poly([0] * i + [1]), Tlist[i])