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
def test_det_and_ortho(self):
    xs = [[ortho_group.rvs(dim) for i in range(10)] for dim in range(2, 12)]
    dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
    assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)
    for xx in xs:
        for x in xx:
            assert_array_almost_equal(np.dot(x, x.T), np.eye(x.shape[0]))