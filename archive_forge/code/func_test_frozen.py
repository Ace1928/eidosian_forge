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
def test_frozen(self):
    rng = np.random.default_rng(28469824356873456)
    alpha = rng.uniform(0, 100, 10)
    x = rng.integers(0, 10, 10)
    n = np.sum(x, axis=-1)
    d = dirichlet_multinomial(alpha, n)
    assert_equal(d.logpmf(x), dirichlet_multinomial.logpmf(x, alpha, n))
    assert_equal(d.pmf(x), dirichlet_multinomial.pmf(x, alpha, n))
    assert_equal(d.mean(), dirichlet_multinomial.mean(alpha, n))
    assert_equal(d.var(), dirichlet_multinomial.var(alpha, n))
    assert_equal(d.cov(), dirichlet_multinomial.cov(alpha, n))