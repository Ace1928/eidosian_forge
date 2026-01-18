from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legline_zeroscl(self):
    assert_equal(leg.legline(3, 0), [3])