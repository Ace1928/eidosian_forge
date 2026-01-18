import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
@pytest.mark.parametrize('test_func,expected', [(stats.circmean, {None: 359.4178026893944, 0: np.array([353.0, 6.0, 3.0, 355.5, 9.5, 349.5]), 1: np.array([0.16769015, 358.66510252])}), (stats.circvar, {None: 0.008396678483192477, 0: np.array([1.9997969, 0.4999873, 0.4999873, 6.1230956, 0.1249992, 0.1249992]) * (np.pi / 180) ** 2, 1: np.array([0.006455174270186603, 0.01016767581393285])}), (stats.circstd, {None: 7.440570778057074, 0: np.array([2.00020313, 1.00002539, 1.00002539, 3.50108929, 0.50000317, 0.50000317]), 1: np.array([6.52070212, 8.19138093])})])
def test_nan_omit_array(self, test_func, expected):
    x = np.array([[355, 5, 2, 359, 10, 350, np.nan], [351, 7, 4, 352, 9, 349, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    for axis in expected.keys():
        out = test_func(x, high=360, nan_policy='omit', axis=axis)
        if axis is None:
            assert_allclose(out, expected[axis], rtol=1e-07)
        else:
            assert_allclose(out[:-1], expected[axis], rtol=1e-07)
            assert_(np.isnan(out[-1]))