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
def test_cho_inv_batch(self):
    """Regression test for gh-8844."""
    a0 = np.array([[2, 1, 0, 0.5], [1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 2]])
    a1 = np.array([[2, -1, 0, 0.5], [-1, 2, 0.5, 0.5], [0, 0.5, 3, 1], [0.5, 0.5, 1, 4]])
    a = np.array([a0, a1])
    ainv = a.copy()
    _cho_inv_batch(ainv)
    ident = np.eye(4)
    assert_allclose(a[0].dot(ainv[0]), ident, atol=1e-15)
    assert_allclose(a[1].dot(ainv[1]), ident, atol=1e-15)