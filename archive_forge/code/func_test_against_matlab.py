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
def test_against_matlab(self):
    rng = np.random.default_rng(2967390923)
    cov = np.array([[6.21786909, 0.2333667, 7.95506077], [0.2333667, 29.67390923, 16.53946426], [7.95506077, 16.53946426, 19.17725252]])
    df = 1.9559939787727658
    dist = stats.multivariate_t(shape=cov, df=df)
    res = dist.cdf([0, 0, 0], random_state=rng)
    ref = 0.2523
    assert_allclose(res, ref, rtol=0.001)