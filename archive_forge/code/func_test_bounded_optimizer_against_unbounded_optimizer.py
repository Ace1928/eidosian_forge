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
def test_bounded_optimizer_against_unbounded_optimizer(self):
    _, lmbda = stats.boxcox(_boxcox_data, lmbda=None)
    bounds = (lmbda + 0.1, lmbda + 1)
    options = {'xatol': 1e-12}

    def optimizer(fun):
        return optimize.minimize_scalar(fun, bounds=bounds, method='bounded', options=options)
    _, lmbda_bounded = stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)
    assert lmbda_bounded != lmbda
    assert_allclose(lmbda_bounded, bounds[0])