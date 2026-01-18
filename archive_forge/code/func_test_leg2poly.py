from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_leg2poly(self):
    for i in range(10):
        assert_almost_equal(leg.leg2poly([0] * i + [1]), Llist[i])