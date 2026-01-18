from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebdomain(self):
    assert_equal(cheb.chebdomain, [-1, 1])