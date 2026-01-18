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
def test_interpolation(self):
    x = np.arange(5)
    y = np.arange(4)
    assert_equal(stats.iqr(x), 2)
    assert_equal(stats.iqr(y), 1.5)
    assert_equal(stats.iqr(x, interpolation='linear'), 2)
    assert_equal(stats.iqr(y, interpolation='linear'), 1.5)
    assert_equal(stats.iqr(x, interpolation='higher'), 2)
    assert_equal(stats.iqr(x, rng=(25, 80), interpolation='higher'), 3)
    assert_equal(stats.iqr(y, interpolation='higher'), 2)
    assert_equal(stats.iqr(x, interpolation='lower'), 2)
    assert_equal(stats.iqr(x, rng=(25, 80), interpolation='lower'), 2)
    assert_equal(stats.iqr(y, interpolation='lower'), 2)
    assert_equal(stats.iqr(x, interpolation='nearest'), 2)
    assert_equal(stats.iqr(y, interpolation='nearest'), 1)
    assert_equal(stats.iqr(x, interpolation='midpoint'), 2)
    assert_equal(stats.iqr(x, rng=(25, 80), interpolation='midpoint'), 2.5)
    assert_equal(stats.iqr(y, interpolation='midpoint'), 2)
    for method in ('inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'median_unbiased', 'normal_unbiased'):
        stats.iqr(y, interpolation=method)
    assert_raises(ValueError, stats.iqr, x, interpolation='foobar')