from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_kde_bandwidth_method_weighted():

    def scotts_factor(kde_obj):
        """Same as default, just check that it works."""
        return np.power(kde_obj.neff, -1.0 / (kde_obj.d + 4))
    np.random.seed(8765678)
    n_basesample = 50
    xn = np.random.randn(n_basesample)
    gkde = stats.gaussian_kde(xn)
    gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
    gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)
    xs = np.linspace(-7, 7, 51)
    kdepdf = gkde.evaluate(xs)
    kdepdf2 = gkde2.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf2)
    kdepdf3 = gkde3.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf3)
    assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')