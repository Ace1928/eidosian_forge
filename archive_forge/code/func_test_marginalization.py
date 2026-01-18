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
def test_marginalization(self):
    mean = np.array([2.5, 3.5])
    cov = np.array([[0.5, 0.2], [0.2, 0.6]])
    n = 2 ** 8 + 1
    delta = 6 / (n - 1)
    v = np.linspace(0, 6, n)
    xv, yv = np.meshgrid(v, v)
    pos = np.empty((n, n, 2))
    pos[:, :, 0] = xv
    pos[:, :, 1] = yv
    pdf = multivariate_normal.pdf(pos, mean, cov)
    margin_x = romb(pdf, delta, axis=0)
    margin_y = romb(pdf, delta, axis=1)
    gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)
    gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)
    assert_allclose(margin_x, gauss_x, rtol=0.01, atol=0.01)
    assert_allclose(margin_y, gauss_y, rtol=0.01, atol=0.01)