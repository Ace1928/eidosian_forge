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
@pytest.mark.parametrize('dim', (3, 7))
def test_against_analytical(self, dim):
    rng = np.random.default_rng(413722918996573)
    A = scipy.linalg.toeplitz(c=[1] + [0.5] * (dim - 1))
    res = stats.multivariate_t(shape=A).cdf([0] * dim, random_state=rng)
    ref = 1 / (dim + 1)
    assert_allclose(res, ref, rtol=5e-05)