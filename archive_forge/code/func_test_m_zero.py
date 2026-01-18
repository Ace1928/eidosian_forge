import numpy as np
import scipy.special as sc
from numpy.testing import assert_almost_equal, assert_array_equal
def test_m_zero(self):
    val = sc.pdtrc([0, 1, 2], 0.0)
    assert_array_equal(val, [0, 0, 0])