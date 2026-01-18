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
def test_cdf_signs(self):
    mean = np.zeros(3)
    cov = np.eye(3)
    df = 10
    b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
    a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
    expected_signs = np.array([1, -1, -1, 1])
    cdf = multivariate_normal.cdf(b, mean, cov, df, lower_limit=a)
    assert_allclose(cdf, cdf[0] * expected_signs)