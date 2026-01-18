from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebtrim(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, cheb.chebtrim, coef, -1)
    assert_equal(cheb.chebtrim(coef), coef[:-1])
    assert_equal(cheb.chebtrim(coef, 1), coef[:-3])
    assert_equal(cheb.chebtrim(coef, 2), [0])