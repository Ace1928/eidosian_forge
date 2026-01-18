from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legdomain(self):
    assert_equal(leg.legdomain, [-1, 1])