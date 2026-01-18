import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_multisample_BCa_against_R():
    x = [0.75859206, 0.5910282, -0.4419409, -0.36654601, 0.34955357, -1.38835871, 0.76735821]
    y = [1.41186073, 0.49775975, 0.08275588, 0.24086388, 0.03567057, 0.52024419, 0.31966611, 1.32067634]

    def statistic(x, y, axis):
        s1 = stats.skew(x, axis=axis)
        s2 = stats.skew(y, axis=axis)
        return s1 - s2
    rng = np.random.default_rng(468865032284792692)
    res_basic = stats.bootstrap((x, y), statistic, method='basic', batch=100, random_state=rng)
    res_percent = stats.bootstrap((x, y), statistic, method='percentile', batch=100, random_state=rng)
    res_bca = stats.bootstrap((x, y), statistic, method='bca', batch=100, random_state=rng)
    mid_basic = np.mean(res_basic.confidence_interval)
    mid_percent = np.mean(res_percent.confidence_interval)
    mid_bca = np.mean(res_bca.confidence_interval)
    mid_wboot = -1.5519
    diff_basic = (mid_basic - mid_wboot) / abs(mid_wboot)
    diff_percent = (mid_percent - mid_wboot) / abs(mid_wboot)
    diff_bca = (mid_bca - mid_wboot) / abs(mid_wboot)
    assert diff_basic < -0.15
    assert diff_percent > 0.15
    assert abs(diff_bca) < 0.03