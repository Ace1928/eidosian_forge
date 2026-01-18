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
def test_covariance_expansion(self):
    num_rows = 4
    num_cols = 3
    M = np.full((num_rows, num_cols), 0.3)
    Uv = np.full(num_rows, 0.2)
    Us = 0.2
    Vv = np.full(num_cols, 0.1)
    Vs = 0.1
    Ir = np.identity(num_rows)
    Ic = np.identity(num_cols)
    assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).rowcov, 0.2 * Ir)
    assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).colcov, 0.1 * Ic)
    assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).rowcov, 0.2 * Ir)
    assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).colcov, 0.1 * Ic)