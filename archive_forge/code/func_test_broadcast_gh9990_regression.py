import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def test_broadcast_gh9990_regression():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([8, 16, 1, 32, 1, 48])
    ans = [stats.reciprocal.cdf(7, _a, _b) for _a, _b in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(7, a, b), ans)
    ans = [stats.reciprocal.cdf(1, _a, _b) for _a, _b in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(1, a, b), ans)
    ans = [stats.reciprocal.cdf(_a, _a, _b) for _a, _b in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(a, a, b), ans)
    ans = [stats.reciprocal.cdf(_b, _a, _b) for _a, _b in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(b, a, b), ans)