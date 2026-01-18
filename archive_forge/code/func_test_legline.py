from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legline(self):
    assert_equal(leg.legline(3, 4), [3, 4])