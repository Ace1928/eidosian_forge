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
@pytest.mark.parametrize('kwds', [{'permutations': 200, 'random_state': 0}, {'trim': 0.2}, {}], ids=['trim', 'permutations', 'basic'])
@pytest.mark.parametrize('axis', [-1, 0])
def test_nans_on_axis(self, kwds, axis):
    a = np.random.randint(10, size=(5, 3, 10)).astype('float')
    b = np.random.randint(10, size=(5, 3, 10)).astype('float')
    a[0][2][3] = np.nan
    b[2][0][6] = np.nan
    expected = np.isnan(np.sum(a + b, axis=axis))
    with suppress_warnings() as sup, np.errstate(invalid='ignore'):
        sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
        sup.filter(RuntimeWarning, 'Precision loss occurred')
        res = stats.ttest_ind(a, b, axis=axis, **kwds)
    p_nans = np.isnan(res.pvalue)
    assert_array_equal(p_nans, expected)
    statistic_nans = np.isnan(res.statistic)
    assert_array_equal(statistic_nans, expected)