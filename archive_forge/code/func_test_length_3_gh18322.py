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
def test_length_3_gh18322(self):
    res = stats.shapiro([0.6931471805599453, 0.0, 0.0])
    assert res.pvalue >= 0
    x = [-0.7746653110021126, -0.4344432067942129, 1.8157053280290931]
    res = stats.shapiro(x)
    assert_allclose(res.statistic, 0.84658770645509)
    assert_allclose(res.pvalue, 0.2313666489882, rtol=1e-06)