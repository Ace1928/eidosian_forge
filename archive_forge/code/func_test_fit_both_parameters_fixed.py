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
def test_fit_both_parameters_fixed(self):
    data = np.full((2, 1), 3)
    mean_fixed = 1.0
    cov_fixed = np.atleast_2d(1.0)
    mean, cov = multivariate_normal.fit(data, fix_mean=mean_fixed, fix_cov=cov_fixed)
    assert_equal(mean, mean_fixed)
    assert_equal(cov, cov_fixed)