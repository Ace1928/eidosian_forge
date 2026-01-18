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
@pytest.mark.parametrize('method_name', ['mean', 'var', 'cov'])
def test_broadcasting_moments(self, method_name):
    alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
    n = np.array([[6], [7], [8]])
    method = getattr(dirichlet_multinomial, method_name)
    res = method(alpha, n)
    assert res.shape == (3, 4, 3) if method_name != 'cov' else (3, 4, 3, 3)
    for j in range(len(n)):
        for k in range(len(alpha)):
            res_ijk = res[j, k]
            ref = method(alpha[k].squeeze(), n[j].squeeze())
            assert_allclose(res_ijk, ref)