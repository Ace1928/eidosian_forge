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
@pytest.mark.parametrize('kappa', [0, 0.0])
def test_kappa_zero(self, kappa):
    msg = "For 'kappa=0' the von Mises-Fisher distribution becomes the uniform distribution on the sphere surface. Consider using 'scipy.stats.uniform_direction' instead."
    with pytest.raises(ValueError, match=msg):
        vonmises_fisher([1, 0], kappa)