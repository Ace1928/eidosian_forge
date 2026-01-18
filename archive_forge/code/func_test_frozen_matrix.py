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
def test_frozen_matrix(self):
    dim = 7
    frozen = unitary_group(dim)
    frozen_seed = unitary_group(dim, seed=514)
    rvs1 = frozen.rvs(random_state=514)
    rvs2 = unitary_group.rvs(dim, random_state=514)
    rvs3 = frozen_seed.rvs(size=1)
    assert_equal(rvs1, rvs2)
    assert_equal(rvs1, rvs3)