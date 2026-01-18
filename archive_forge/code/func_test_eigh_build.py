import warnings
import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
def test_eigh_build(self):
    rvals = [68.60568999, 89.57756725, 106.67185574]
    cov = array([[77.70273908, 3.51489954, 15.64602427], [3.51489954, 88.97013878, -1.07431931], [15.64602427, -1.07431931, 98.18223512]])
    vals, vecs = linalg.eigh(cov)
    assert_array_almost_equal(vals, rvals)