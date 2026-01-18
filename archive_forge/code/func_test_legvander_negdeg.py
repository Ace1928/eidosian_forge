from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legvander_negdeg(self):
    assert_raises(ValueError, leg.legvander, (1, 2, 3), -1)