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
def test_against_TileStats(self):
    ps = [0.005, 0.009, 0.019, 0.022, 0.051, 0.101, 0.361, 0.387]
    res = stats.false_discovery_control(ps)
    ref = [0.036, 0.036, 0.044, 0.044, 0.082, 0.135, 0.387, 0.387]
    assert_allclose(res, ref, atol=0.001)