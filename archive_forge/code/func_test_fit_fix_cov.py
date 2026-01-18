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
def test_fit_fix_cov(self):
    rng = np.random.default_rng(4385269356937404)
    loc = rng.random(3)
    A = rng.random((3, 3))
    cov = np.dot(A, A.T)
    samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100, random_state=rng)
    mean_free, cov_free = multivariate_normal.fit(samples)
    logp_free = multivariate_normal.logpdf(samples, mean=mean_free, cov=cov_free).sum()
    mean_fix, cov_fix = multivariate_normal.fit(samples, fix_cov=cov)
    assert_equal(mean_fix, np.mean(samples, axis=0))
    assert_equal(cov_fix, cov)
    logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix, cov=cov_fix).sum()
    assert logp_fix < logp_free
    mean_perturbed = mean_fix + 1e-08 * rng.random(3)
    logp_perturbed = multivariate_normal.logpdf(samples, mean=mean_perturbed, cov=cov_fix).sum()
    assert logp_perturbed < logp_fix