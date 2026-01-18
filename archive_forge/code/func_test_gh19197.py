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
def test_gh19197(self):
    mean = np.ones(2)
    cov = Covariance.from_eigendecomposition((np.zeros(2), np.eye(2)))
    dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    rvs = dist.rvs(size=None)
    assert_equal(rvs, mean)
    cov = scipy.stats.Covariance.from_eigendecomposition((np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 400.0]])))
    dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    rvs = dist.rvs(size=None)
    assert rvs[0] != mean[0]
    assert rvs[1] == mean[1]