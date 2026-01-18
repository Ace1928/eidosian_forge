import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_array_smoothing(self):
    rng = np.random.RandomState(0)
    seq = Halton(1, scramble=False, seed=rng)
    degree = 2
    x = seq.random(50)
    P = _vandermonde(x, degree)
    poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
    y = P.dot(poly_coeffs)
    y_with_outlier = np.copy(y)
    y_with_outlier[10] += 1.0
    smoothing = np.zeros((50,))
    smoothing[10] = 1000.0
    yitp = self.build(x, y_with_outlier, smoothing=smoothing)(x)
    assert_allclose(yitp, y, atol=0.0001)