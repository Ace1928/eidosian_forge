import os
import re
import warnings
from collections import namedtuple
from itertools import product
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
from numpy.lib import NumpyVersion
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
@pytest.mark.parametrize('alternative', ['two-sided', 'greater', 'less'])
def test_binom_test_deprecation(alternative):
    deprecation_msg = "'binom_test' is deprecated in favour of 'binomtest' from version 1.7.0 and will be removed in Scipy 1.12.0."
    num = 10
    rng = np.random.default_rng(156114182869662948677852568516310985853)
    X = rng.integers(10, 100, (num,))
    N = X + rng.integers(0, 100, (num,))
    P = rng.uniform(0, 1, (num,))
    for x, n, p in zip(X, N, P):
        with pytest.warns(DeprecationWarning, match=deprecation_msg):
            res = stats.binom_test(x, n, p, alternative=alternative)
        assert res == stats.binomtest(x, n, p, alternative=alternative).pvalue