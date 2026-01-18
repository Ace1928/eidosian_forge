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
def test_matches_multivariate(self):
    for i in range(1, 5):
        for j in range(1, 5):
            M = np.full((i, j), 0.3)
            U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
            V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
            frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
            X = frozen.rvs(random_state=1234)
            pdf1 = frozen.pdf(X)
            logpdf1 = frozen.logpdf(X)
            entropy1 = frozen.entropy()
            vecX = X.T.flatten()
            vecM = M.T.flatten()
            cov = np.kron(V, U)
            pdf2 = multivariate_normal.pdf(vecX, mean=vecM, cov=cov)
            logpdf2 = multivariate_normal.logpdf(vecX, mean=vecM, cov=cov)
            entropy2 = multivariate_normal.entropy(mean=vecM, cov=cov)
            assert_allclose(pdf1, pdf2, rtol=1e-10)
            assert_allclose(logpdf1, logpdf2, rtol=1e-10)
            assert_allclose(entropy1, entropy2)