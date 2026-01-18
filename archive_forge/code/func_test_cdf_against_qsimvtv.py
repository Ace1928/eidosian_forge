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
@pytest.mark.parametrize('dim', [2, 3, 5, 10])
@pytest.mark.parametrize('seed', [3363958638, 7891119608, 3887698049, 5013150848, 1495033423, 6170824608])
@pytest.mark.parametrize('singular', [False, True])
def test_cdf_against_qsimvtv(self, dim, seed, singular):
    if singular and seed != 3363958638:
        pytest.skip('Agreement with qsimvtv is not great in singular case')
    rng = np.random.default_rng(seed)
    w = 10 ** rng.uniform(-2, 2, size=dim)
    cov = _random_covariance(dim, w, rng, singular)
    mean = rng.random(dim)
    a = -rng.random(dim)
    b = rng.random(dim)
    df = rng.random() * 5
    res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, allow_singular=True)
    with np.errstate(invalid='ignore'):
        ref = _qsimvtv(20000, df, cov, np.inf * a, b - mean, rng)[0]
    assert_allclose(res, ref, atol=0.0002, rtol=0.001)
    res = stats.multivariate_t.cdf(b, mean, cov, df, lower_limit=a, random_state=rng, allow_singular=True)
    with np.errstate(invalid='ignore'):
        ref = _qsimvtv(20000, df, cov, a - mean, b - mean, rng)[0]
    assert_allclose(res, ref, atol=0.0001, rtol=0.001)