import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
@pytest.mark.parametrize('alpha', [0.2, 0.5 - 1e-12, 0.5, 0.5 + 1e-12, 0.8])
@pytest.mark.parametrize('n', [20, 2000])
def test_expectile_properties(self, alpha, n):
    """
        See Section 6 of
        I. Steinwart, C. Pasin, R.C. Williamson & S. Zhang (2014).
        "Elicitation and Identification of Properties". COLT.
        http://proceedings.mlr.press/v35/steinwart14.html

        and

        Propositions 5, 6, 7 of
        F. Bellini, B. Klar, and A. Müller and E. Rosazza Gianin (2013).
        "Generalized Quantiles as Risk Measures"
        http://doi.org/10.2139/ssrn.2225751
        """
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    for c in [-5, 0, 0.5]:
        assert_allclose(stats.expectile(np.full(shape=n, fill_value=c), alpha=alpha), c)
    c = rng.exponential()
    assert_allclose(stats.expectile(x + c, alpha=alpha), stats.expectile(x, alpha=alpha) + c)
    assert_allclose(stats.expectile(x - c, alpha=alpha), stats.expectile(x, alpha=alpha) - c)
    assert_allclose(stats.expectile(c * x, alpha=alpha), c * stats.expectile(x, alpha=alpha))
    y = rng.logistic(size=n, loc=10)
    if alpha == 0.5:

        def assert_op(a, b):
            assert_allclose(a, b)
    elif alpha > 0.5:

        def assert_op(a, b):
            assert a < b
    else:

        def assert_op(a, b):
            assert a > b
    assert_op(stats.expectile(np.r_[x + y], alpha=alpha), stats.expectile(x, alpha=alpha) + stats.expectile(y, alpha=alpha))
    y = rng.normal(size=n, loc=5)
    assert stats.expectile(x, alpha=alpha) <= stats.expectile(y, alpha=alpha)
    y = rng.logistic(size=n, loc=10)
    for c in [0.1, 0.5, 0.8]:
        assert_op(stats.expectile((1 - c) * x + c * y, alpha=alpha), (1 - c) * stats.expectile(x, alpha=alpha) + c * stats.expectile(y, alpha=alpha))
    assert_allclose(stats.expectile(-x, alpha=alpha), -stats.expectile(x, alpha=1 - alpha))