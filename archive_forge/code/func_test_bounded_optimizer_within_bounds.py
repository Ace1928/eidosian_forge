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
@pytest.mark.parametrize('method', ['mle', 'pearsonr', 'all'])
@pytest.mark.parametrize('bounds', [(-1, 1), (1.1, 2), (-2, -1.1)])
def test_bounded_optimizer_within_bounds(self, method, bounds):

    def optimizer(fun):
        return optimize.minimize_scalar(fun, bounds=bounds, method='bounded')
    maxlog = stats.boxcox_normmax(self.x, method=method, optimizer=optimizer)
    assert np.all(bounds[0] < maxlog)
    assert np.all(maxlog < bounds[1])