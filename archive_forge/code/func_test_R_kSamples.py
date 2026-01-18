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
def test_R_kSamples(self):
    x1 = np.linspace(1, 100, 100)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='p-value floored')
        s, _, p = stats.anderson_ksamp([x1, x1 + 40.5], midrank=False)
    assert_almost_equal(s, 41.105, 3)
    assert_equal(p, 0.001)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='p-value floored')
        s, _, p = stats.anderson_ksamp([x1, x1 + 40.5])
    assert_almost_equal(s, 41.235, 3)
    assert_equal(p, 0.001)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='p-value capped')
        s, _, p = stats.anderson_ksamp([x1, x1 + 0.5], midrank=False)
    assert_almost_equal(s, -1.2824, 4)
    assert_equal(p, 0.25)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='p-value capped')
        s, _, p = stats.anderson_ksamp([x1, x1 + 0.5])
    assert_almost_equal(s, -1.2944, 4)
    assert_equal(p, 0.25)
    s, _, p = stats.anderson_ksamp([x1, x1 + 7.5], midrank=False)
    assert_almost_equal(s, 1.4923, 4)
    assert_allclose(p, 0.0775, atol=0.005, rtol=0)
    s, _, p = stats.anderson_ksamp([x1, x1 + 6])
    assert_almost_equal(s, 0.6389, 4)
    assert_allclose(p, 0.1798, atol=0.005, rtol=0)
    s, _, p = stats.anderson_ksamp([x1, x1 + 11.5], midrank=False)
    assert_almost_equal(s, 4.5042, 4)
    assert_allclose(p, 0.00545, atol=0.0005, rtol=0)
    s, _, p = stats.anderson_ksamp([x1, x1 + 13.5], midrank=False)
    assert_almost_equal(s, 6.2982, 4)
    assert_allclose(p, 0.00118, atol=0.0001, rtol=0)