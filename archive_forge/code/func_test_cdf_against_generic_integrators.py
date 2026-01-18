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
def test_cdf_against_generic_integrators(self):
    dim = 3
    rng = np.random.default_rng(41372291899657)
    w = 10 ** rng.uniform(-1, 1, size=dim)
    cov = _random_covariance(dim, w, rng, singular=True)
    mean = rng.random(dim)
    a = -rng.random(dim)
    b = rng.random(dim)
    df = rng.random() * 5
    res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng, lower_limit=a)

    def integrand(x):
        return stats.multivariate_t.pdf(x.T, mean, cov, df)
    ref = qmc_quad(integrand, a, b, qrng=stats.qmc.Halton(d=dim, seed=rng))
    assert_allclose(res, ref.integral, rtol=0.001)

    def integrand(*zyx):
        return stats.multivariate_t.pdf(zyx[::-1], mean, cov, df)
    ref = tplquad(integrand, a[0], b[0], a[1], b[1], a[2], b[2])
    assert_allclose(res, ref[0], rtol=0.001)