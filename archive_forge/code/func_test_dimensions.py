from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_dimensions(self):
    for i in range(1, 5):
        coef = [0] * i + [1]
        assert_(leg.legcompanion(coef).shape == (i, i))