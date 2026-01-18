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
@pytest.mark.parametrize('cov_type_name', _all_covariance_types[:-1])
def test_factories(self, cov_type_name):
    A = np.diag([1, 2, 3])
    x = [-4, 2, 5]
    cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
    preprocessing = self._covariance_preprocessing[cov_type_name]
    factory = getattr(Covariance, f'from_{cov_type_name.lower()}')
    res = factory(preprocessing(A))
    ref = cov_type(preprocessing(A))
    assert type(res) == type(ref)
    assert_allclose(res.whiten(x), ref.whiten(x))