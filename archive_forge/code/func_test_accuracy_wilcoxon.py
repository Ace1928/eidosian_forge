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
def test_accuracy_wilcoxon(self):
    freq = [1, 4, 16, 15, 8, 4, 5, 1, 2]
    nums = range(-4, 5)
    x = np.concatenate([[u] * v for u, v in zip(nums, freq)])
    y = np.zeros(x.size)
    T, p = stats.wilcoxon(x, y, 'pratt', mode='approx')
    assert_allclose(T, 423)
    assert_allclose(p, 0.0031724568006762576)
    T, p = stats.wilcoxon(x, y, 'zsplit', mode='approx')
    assert_allclose(T, 441)
    assert_allclose(p, 0.0032145343172473055)
    T, p = stats.wilcoxon(x, y, 'wilcox', mode='approx')
    assert_allclose(T, 327)
    assert_allclose(p, 0.00641346115861)
    x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
    y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
    T, p = stats.wilcoxon(x, y, correction=False, mode='approx')
    assert_equal(T, 34)
    assert_allclose(p, 0.6948866, rtol=1e-06)
    T, p = stats.wilcoxon(x, y, correction=True, mode='approx')
    assert_equal(T, 34)
    assert_allclose(p, 0.7240817, rtol=1e-06)