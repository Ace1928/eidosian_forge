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
def test_cdf_with_lower_limit_consistency(self):
    rng = np.random.default_rng(2408071309372769818)
    mean = rng.random(3)
    cov = rng.random((3, 3))
    cov = cov @ cov.T
    a = rng.random((2, 3)) * 6 - 3
    b = rng.random((2, 3)) * 6 - 3
    cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
    cdf2 = multivariate_normal(mean, cov).cdf(b, lower_limit=a)
    cdf3 = np.exp(multivariate_normal.logcdf(b, mean, cov, lower_limit=a))
    cdf4 = np.exp(multivariate_normal(mean, cov).logcdf(b, lower_limit=a))
    assert_allclose(cdf2, cdf1, rtol=0.0001)
    assert_allclose(cdf3, cdf1, rtol=0.0001)
    assert_allclose(cdf4, cdf1, rtol=0.0001)