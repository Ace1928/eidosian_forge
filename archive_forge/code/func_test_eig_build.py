import warnings
import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
def test_eig_build(self):
    rva = array([103.221168 + 0j, -19.1843603 + 0j, -0.604004526 + 15.84422474j, -0.604004526 - 15.84422474j, -11.3692929 + 0j, -0.657612485 + 10.41755503j, -0.657612485 - 10.41755503j, 18.2126812 + 0j, 10.6011014 + 0j, 7.80732773 + 0j, -0.765390898 + 0j, 1.51971555e-15 + 0j, -1.51308713e-15 + 0j])
    a = arange(13 * 13, dtype=float64)
    a.shape = (13, 13)
    a = a % 17
    va, ve = linalg.eig(a)
    va.sort()
    rva.sort()
    assert_array_almost_equal(va, rva)