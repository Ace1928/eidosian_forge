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
def test_friedmanchisquare():
    x1 = [array([0.763, 0.599, 0.954, 0.628, 0.882, 0.936, 0.661, 0.583, 0.775, 1.0, 0.94, 0.619, 0.972, 0.957]), array([0.768, 0.591, 0.971, 0.661, 0.888, 0.931, 0.668, 0.583, 0.838, 1.0, 0.962, 0.666, 0.981, 0.978]), array([0.771, 0.59, 0.968, 0.654, 0.886, 0.916, 0.609, 0.563, 0.866, 1.0, 0.965, 0.614, 0.9751, 0.946]), array([0.798, 0.569, 0.967, 0.657, 0.898, 0.931, 0.685, 0.625, 0.875, 1.0, 0.962, 0.669, 0.975, 0.97])]
    x2 = [array([4, 3, 5, 3, 5, 3, 2, 5, 4, 4, 4, 3]), array([2, 2, 1, 2, 3, 1, 2, 3, 2, 1, 1, 3]), array([2, 4, 3, 3, 4, 3, 3, 4, 4, 1, 2, 1]), array([3, 5, 4, 3, 4, 4, 3, 3, 3, 4, 4, 4])]
    x3 = [array([7.0, 9.9, 8.5, 5.1, 10.3]), array([5.3, 5.7, 4.7, 3.5, 7.7]), array([4.9, 7.6, 5.5, 2.8, 8.4]), array([8.8, 8.9, 8.1, 3.3, 9.1])]
    assert_array_almost_equal(stats.friedmanchisquare(x1[0], x1[1], x1[2], x1[3]), (10.2283464566929, 0.0167215803284414))
    assert_array_almost_equal(stats.friedmanchisquare(x2[0], x2[1], x2[2], x2[3]), (18.9428571428571, 0.000280938375189499))
    assert_array_almost_equal(stats.friedmanchisquare(x3[0], x3[1], x3[2], x3[3]), (10.68, 0.0135882729582176))
    assert_raises(ValueError, stats.friedmanchisquare, x3[0], x3[1])
    attributes = ('statistic', 'pvalue')
    res = stats.friedmanchisquare(*x1)
    check_named_results(res, attributes)
    assert_array_almost_equal(mstats.friedmanchisquare(x1[0], x1[1], x1[2], x1[3]), (10.2283464566929, 0.0167215803284414))
    assert_array_almost_equal(mstats.friedmanchisquare(x3[0], x3[1], x3[2], x3[3]), (10.68, 0.0135882729582176))
    assert_raises(ValueError, mstats.friedmanchisquare, x3[0], x3[1])