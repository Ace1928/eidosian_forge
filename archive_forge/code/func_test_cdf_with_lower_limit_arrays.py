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
def test_cdf_with_lower_limit_arrays(self):
    rng = np.random.default_rng(2408071309372769818)
    mean = [0, 0]
    cov = np.eye(2)
    a = rng.random((4, 3, 2)) * 6 - 3
    b = rng.random((4, 3, 2)) * 6 - 3
    cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
    cdf2a = multivariate_normal.cdf(b, mean, cov)
    cdf2b = multivariate_normal.cdf(a, mean, cov)
    ab1 = np.concatenate((a[..., 0:1], b[..., 1:2]), axis=-1)
    ab2 = np.concatenate((a[..., 1:2], b[..., 0:1]), axis=-1)
    cdf2ab1 = multivariate_normal.cdf(ab1, mean, cov)
    cdf2ab2 = multivariate_normal.cdf(ab2, mean, cov)
    cdf2 = cdf2a + cdf2b - cdf2ab1 - cdf2ab2
    assert_allclose(cdf1, cdf2)