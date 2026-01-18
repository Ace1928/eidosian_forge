import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
def test_K_and_K_minus_1_calls_equal(self):
    np.random.seed(2846)
    n = np.random.randint(1, 32)
    alpha = np.random.uniform(1e-09, 100, n)
    d = dirichlet(alpha)
    num_tests = 10
    for i in range(num_tests):
        x = np.random.uniform(1e-09, 100, n)
        x /= np.sum(x)
        assert_almost_equal(d.pdf(x[:-1]), d.pdf(x))