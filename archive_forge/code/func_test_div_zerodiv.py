import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_div_zerodiv(self):
    assert_raises(ZeroDivisionError, pu._div, pu._div, (1, 2, 3), [0])