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
def test_array_input(self):
    num_rows = 4
    num_cols = 3
    M = np.full((num_rows, num_cols), 0.3)
    U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
    V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
    N = 10
    frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
    X1 = frozen.rvs(size=N, random_state=1234)
    X2 = frozen.rvs(size=N, random_state=4321)
    X = np.concatenate((X1[np.newaxis, :, :, :], X2[np.newaxis, :, :, :]), axis=0)
    assert_equal(X.shape, (2, N, num_rows, num_cols))
    array_logpdf = frozen.logpdf(X)
    assert_equal(array_logpdf.shape, (2, N))
    for i in range(2):
        for j in range(N):
            separate_logpdf = matrix_normal.logpdf(X[i, j], mean=M, rowcov=U, colcov=V)
            assert_allclose(separate_logpdf, array_logpdf[i, j], 1e-10)