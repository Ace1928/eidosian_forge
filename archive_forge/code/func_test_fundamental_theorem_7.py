import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
@pytest.mark.slow
def test_fundamental_theorem_7(self):
    self.fundamental_theorem(7, 0.5, 1.2)