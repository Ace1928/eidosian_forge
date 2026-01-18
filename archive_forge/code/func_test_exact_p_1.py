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
@pytest.mark.parametrize('x', [[-1, -2, 3], [-1, 2, -3, -4, 5], [-1, -2, 3, -4, -5, -6, 7, 8]])
def test_exact_p_1(self, x):
    w, p = stats.wilcoxon(x)
    x = np.array(x)
    wtrue = x[x > 0].sum()
    assert_equal(w, wtrue)
    assert_equal(p, 1)