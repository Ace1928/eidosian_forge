from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_gaussian_kde_covariance_caching():
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=5)
    y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754, 0.01664475]
    kde = stats.gaussian_kde(x1)
    kde.set_bandwidth(bw_method=0.5)
    kde.set_bandwidth(bw_method='scott')
    y2 = kde(xs)
    assert_array_almost_equal(y_expected, y2, decimal=7)