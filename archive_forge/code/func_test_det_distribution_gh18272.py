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
@pytest.mark.parametrize('dim', [2, 5, 10, 20])
def test_det_distribution_gh18272(self, dim):
    rng = np.random.default_rng(6796248956179332344)
    dist = ortho_group(dim=dim)
    rvs = dist.rvs(size=5000, random_state=rng)
    dets = scipy.linalg.det(rvs)
    k = np.sum(dets > 0)
    n = len(dets)
    res = stats.binomtest(k, n)
    low, high = res.proportion_ci(confidence_level=0.95)
    assert low < 0.5 < high