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
@patch('scipy.stats.multivariate_normal._logpdf')
def test_mvt_with_inf_df_calls_normal(self, mock):
    dist = multivariate_t(0, 1, df=np.inf, seed=7)
    assert isinstance(dist, multivariate_normal_frozen)
    multivariate_t.pdf(0, df=np.inf)
    assert mock.call_count == 1
    multivariate_t.logpdf(0, df=np.inf)
    assert mock.call_count == 2