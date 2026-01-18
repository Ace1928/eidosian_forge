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
@pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
def test_broadcasting(self, method):
    testshape = (2, 2)
    rng = np.random.default_rng(2777937887058094419)
    x = uniform_direction(3).rvs(testshape, random_state=rng)
    mu = np.full((3,), 1 / np.sqrt(3))
    kappa = 5
    result_all = method(x, mu, kappa)
    assert result_all.shape == testshape
    for i in range(testshape[0]):
        for j in range(testshape[1]):
            current_val = method(x[i, j, :], mu, kappa)
            assert_allclose(current_val, result_all[i, j], rtol=1e-15)