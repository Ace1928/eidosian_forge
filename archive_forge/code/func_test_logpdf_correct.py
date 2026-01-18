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
@pytest.mark.parametrize('x, loc, shape, df, ans', PDF_TESTS)
def test_logpdf_correct(self, x, loc, shape, df, ans):
    dist = multivariate_t(loc, shape, df, seed=0)
    val1 = dist.pdf(x)
    val2 = dist.logpdf(x)
    assert_array_almost_equal(np.log(val1), val2)