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
def test_bad_input(self):
    num_rows = 4
    num_cols = 3
    M = np.full((num_rows, num_cols), 0.3)
    U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
    V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
    assert_raises(ValueError, matrix_normal, np.zeros((5, 4, 3)))
    assert_raises(ValueError, matrix_normal, M, np.zeros(10), V)
    assert_raises(ValueError, matrix_normal, M, U, np.zeros(10))
    assert_raises(ValueError, matrix_normal, M, U, U)
    assert_raises(ValueError, matrix_normal, M, V, V)
    assert_raises(ValueError, matrix_normal, M.T, U, V)
    e = np.linalg.LinAlgError
    assert_raises(e, matrix_normal.rvs, M, U, np.ones((num_cols, num_cols)))
    assert_raises(e, matrix_normal.rvs, M, np.ones((num_rows, num_rows)), V)
    assert_raises(e, matrix_normal, M, U, np.ones((num_cols, num_cols)))
    assert_raises(e, matrix_normal, M, np.ones((num_rows, num_rows)), V)