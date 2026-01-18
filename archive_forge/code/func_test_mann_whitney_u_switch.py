from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
@pytest.mark.xslow
def test_mann_whitney_u_switch():
    _mwu_state._recursive = None
    _mwu_state._fmnks = -np.ones((1, 1, 1))
    rng = np.random.default_rng(9546146887652)
    x = rng.random(5)
    y = rng.random(501)
    stats.mannwhitneyu(x, y, method='exact')
    assert np.all(_mwu_state._fmnks == -1)
    y = rng.random(500)
    stats.mannwhitneyu(x, y, method='exact')
    assert not np.all(_mwu_state._fmnks == -1)