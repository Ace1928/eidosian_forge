import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_fuzz(self):
    rfuncs = (np.random.uniform, np.random.normal, np.random.standard_cauchy, np.random.exponential)
    ntests = 100
    for i in range(ntests):
        rfunc = random.choice(rfuncs)
        target_norm_1 = random.expovariate(1.0)
        n = random.randrange(2, 16)
        A_original = rfunc(size=(n, n))
        E_original = rfunc(size=(n, n))
        A_original_norm_1 = scipy.linalg.norm(A_original, 1)
        scale = target_norm_1 / A_original_norm_1
        A = scale * A_original
        E = scale * E_original
        M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm(M)[:n, n:]
        observed_expm, observed_frechet = expm_frechet(A, E)
        assert_allclose(expected_expm, observed_expm, atol=5e-08)
        assert_allclose(expected_frechet, observed_frechet, atol=1e-07)