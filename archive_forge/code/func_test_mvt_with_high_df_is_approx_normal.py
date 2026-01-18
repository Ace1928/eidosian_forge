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
def test_mvt_with_high_df_is_approx_normal(self):
    P_VAL_MIN = 0.1
    dist = multivariate_t(0, 1, df=100000, seed=1)
    samples = dist.rvs(size=100000)
    _, p = normaltest(samples)
    assert p > P_VAL_MIN
    dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000, seed=42)
    samples = dist.rvs(size=100000)
    _, p = normaltest(samples)
    assert (p > P_VAL_MIN).all()