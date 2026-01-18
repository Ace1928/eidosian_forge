from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_approximation(self):

    def powx(x, p):
        return x ** p
    x = np.linspace(-1, 1, 10)
    for deg in range(0, 10):
        for p in range(0, deg + 1):
            c = cheb.chebinterpolate(powx, deg, (p,))
            assert_almost_equal(cheb.chebval(x, c), powx(x, p), decimal=12)