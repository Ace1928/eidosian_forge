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
def test_1D_is_invgamma(self):
    np.random.seed(482974)
    sn = 500
    dim = 1
    scale = np.eye(dim)
    df_range = np.arange(5, 20, 2, dtype=float)
    X = np.linspace(0.1, 10, num=10)
    for df in df_range:
        iw = invwishart(df, scale)
        ig = invgamma(df / 2, scale=1.0 / 2)
        assert_allclose(iw.var(), ig.var())
        assert_allclose(iw.mean(), ig.mean())
        assert_allclose(iw.pdf(X), ig.pdf(X))
        rvs = iw.rvs(size=sn)
        args = (df / 2, 0, 1.0 / 2)
        alpha = 0.01
        check_distribution_rvs('invgamma', args, alpha, rvs)
        assert_allclose(iw.entropy(), ig.entropy())