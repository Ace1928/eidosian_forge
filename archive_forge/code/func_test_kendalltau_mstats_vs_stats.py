import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
@pytest.mark.parametrize('method', ('exact', 'asymptotic'))
@pytest.mark.parametrize('alternative', ('two-sided', 'greater', 'less'))
def test_kendalltau_mstats_vs_stats(self, method, alternative):
    np.random.seed(0)
    n = 50
    x = np.random.rand(n)
    y = np.random.rand(n)
    mask = np.random.rand(n) > 0.5
    x_masked = ma.array(x, mask=mask)
    y_masked = ma.array(y, mask=mask)
    res_masked = mstats.kendalltau(x_masked, y_masked, method=method, alternative=alternative)
    x_compressed = x_masked.compressed()
    y_compressed = y_masked.compressed()
    res_compressed = stats.kendalltau(x_compressed, y_compressed, method=method, alternative=alternative)
    x[mask] = np.nan
    y[mask] = np.nan
    res_nan = stats.kendalltau(x, y, method=method, nan_policy='omit', alternative=alternative)
    assert_allclose(res_masked, res_compressed)
    assert_allclose(res_nan, res_compressed)