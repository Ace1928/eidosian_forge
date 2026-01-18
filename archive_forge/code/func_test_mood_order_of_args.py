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
def test_mood_order_of_args(self):
    np.random.seed(1234)
    x1 = np.random.randn(10, 1)
    x2 = np.random.randn(15, 1)
    z1, p1 = stats.mood(x1, x2)
    z2, p2 = stats.mood(x2, x1)
    assert_array_almost_equal([z1, p1], [-z2, p2])