from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermdomain(self):
    assert_equal(herm.hermdomain, [-1, 1])