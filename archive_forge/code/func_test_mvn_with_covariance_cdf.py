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
@pytest.mark.parametrize('size', [tuple(), (2, 4, 3)])
@pytest.mark.parametrize('cov_type_name', _all_covariance_types)
def test_mvn_with_covariance_cdf(self, size, cov_type_name):
    matrix_type = 'diagonal full rank'
    A = self._matrices[matrix_type]
    cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
    preprocessing = self._covariance_preprocessing[cov_type_name]
    mean = [0.1, 0.2, 0.3]
    cov_object = cov_type(preprocessing(A))
    mvn = multivariate_normal
    dist0 = multivariate_normal(mean, A, allow_singular=True)
    dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
    rng = np.random.default_rng(5292808890472453840)
    x = rng.multivariate_normal(mean, A, size=size)
    assert_close(mvn.cdf(x, mean, cov_object), dist0.cdf(x))
    assert_close(dist1.cdf(x), dist0.cdf(x))
    assert_close(mvn.logcdf(x, mean, cov_object), dist0.logcdf(x))
    assert_close(dist1.logcdf(x), dist0.logcdf(x))