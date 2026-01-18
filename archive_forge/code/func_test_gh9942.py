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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_gh9942(self):
    A = np.diag([1, 2, -1e-08])
    n = A.shape[0]
    mean = np.zeros(n)
    with pytest.raises(ValueError, match='The input matrix must be...'):
        multivariate_normal(mean, A).rvs()
    seed = 3562050283508273023
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    cov = Covariance.from_eigendecomposition(np.linalg.eigh(A))
    rv = multivariate_normal(mean, cov)
    res = rv.rvs(random_state=rng1)
    ref = multivariate_normal.rvs(mean, cov, random_state=rng2)
    assert_equal(res, ref)