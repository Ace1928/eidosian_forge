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
def test_cov_broadcasting(self):
    cov1 = multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
    cov2 = [[[1.05, -1.05], [-1.05, 1.05]], [[1.56, -1.56], [-1.56, 1.56]]]
    assert_allclose(cov1, cov2, rtol=1e-08)
    cov3 = multivariate_hypergeom.cov(m=[[4], [5]], n=[4, 5])
    cov4 = [[[0.0]], [[0.0]]]
    assert_allclose(cov3, cov4, rtol=1e-08)
    cov5 = multivariate_hypergeom.cov(m=[7, 9], n=[8, 12])
    cov6 = [[[1.05, -1.05], [-1.05, 1.05]], [[0.7875, -0.7875], [-0.7875, 0.7875]]]
    assert_allclose(cov5, cov6, rtol=1e-08)