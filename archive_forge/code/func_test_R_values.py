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
def test_R_values(self):
    r_pdf = np.array([0.0002214706, 0.0013819953, 0.0049138692, 0.010380305, 0.01402508])
    x = np.linspace(0, 2, 5)
    y = 3 * x - 2
    z = x + np.cos(y)
    r = np.array([x, y, z]).T
    mean = np.array([1, 3, 2], 'd')
    cov = np.array([[1, 2, 0], [2, 5, 0.5], [0, 0.5, 3]], 'd')
    pdf = multivariate_normal.pdf(r, mean, cov)
    assert_allclose(pdf, r_pdf, atol=1e-10)
    r_cdf = np.array([0.0017866215, 0.0267142892, 0.0857098761, 0.1063242573, 0.2501068509])
    cdf = multivariate_normal.cdf(r, mean, cov)
    assert_allclose(cdf, r_cdf, atol=2e-05)
    r_cdf2 = np.array([0.01262147, 0.05838989, 0.18389571, 0.40696599, 0.66470577])
    r2 = np.array([x, y]).T
    mean2 = np.array([1, 3], 'd')
    cov2 = np.array([[1, 2], [2, 5]], 'd')
    cdf2 = multivariate_normal.cdf(r2, mean2, cov2)
    assert_allclose(cdf2, r_cdf2, atol=1e-05)