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
def test_alternative_exact(self):
    x1 = [-5, 1, 5, 10, 15, 20, 25]
    x2 = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    statistic, pval = stats.ansari(x1, x2)
    pval_l = stats.ansari(x1, x2, alternative='less').pvalue
    pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
    assert pval_l > 0.95
    assert pval_g < 0.05
    prob = _abw_state.pmf(statistic, len(x1), len(x2))
    assert_allclose(pval_g + pval_l, 1 + prob, atol=1e-12)
    assert_allclose(pval_g, pval / 2, atol=1e-12)
    assert_allclose(pval_l, 1 + prob - pval / 2, atol=1e-12)
    pval_l_reverse = stats.ansari(x2, x1, alternative='less').pvalue
    pval_g_reverse = stats.ansari(x2, x1, alternative='greater').pvalue
    assert pval_l_reverse < 0.05
    assert pval_g_reverse > 0.95