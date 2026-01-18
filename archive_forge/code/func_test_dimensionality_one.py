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
def test_dimensionality_one(self):
    n = 6
    alpha = [10]
    x = np.asarray([n])
    dist = dirichlet_multinomial(alpha, n)
    assert_equal(dist.pmf(x), 1)
    assert_equal(dist.pmf(x + 1), 0)
    assert_equal(dist.logpmf(x), 0)
    assert_equal(dist.logpmf(x + 1), -np.inf)
    assert_equal(dist.mean(), n)
    assert_equal(dist.var(), 0)
    assert_equal(dist.cov(), 0)