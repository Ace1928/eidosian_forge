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
def test_data_with_zeros(self):
    alpha = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([0.1, 0.0, 0.2, 0.7])
    dirichlet.pdf(x, alpha)
    dirichlet.logpdf(x, alpha)
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    assert_almost_equal(dirichlet.pdf(x, alpha), 6)
    assert_almost_equal(dirichlet.logpdf(x, alpha), np.log(6))