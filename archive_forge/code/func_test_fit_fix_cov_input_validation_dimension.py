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
@pytest.mark.parametrize('fix_cov', [np.zeros((2,)), np.zeros((3, 2)), np.zeros((4, 4))])
def test_fit_fix_cov_input_validation_dimension(self, fix_cov):
    msg = '`fix_cov` must be a two-dimensional square array of same side length as the dimensionality of the vectors `x`.'
    with pytest.raises(ValueError, match=msg):
        multivariate_normal.fit(np.eye(3), fix_cov=fix_cov)