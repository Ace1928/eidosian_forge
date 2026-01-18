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
def test_exception_singular_cov(self):
    np.random.seed(1234)
    x = np.random.randn(5)
    mean = np.random.randn(5)
    cov = np.ones((5, 5))
    e = np.linalg.LinAlgError
    assert_raises(e, multivariate_normal, mean, cov)
    assert_raises(e, multivariate_normal.pdf, x, mean, cov)
    assert_raises(e, multivariate_normal.logpdf, x, mean, cov)
    assert_raises(e, multivariate_normal.cdf, x, mean, cov)
    assert_raises(e, multivariate_normal.logcdf, x, mean, cov)
    cov = [[1.0, 0.0], [1.0, 1.0]]
    msg = 'When `allow_singular is False`, the input matrix'
    with pytest.raises(np.linalg.LinAlgError, match=msg):
        multivariate_normal(cov=cov)