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
def test_linregress(self):
    x = np.arange(11)
    y = np.arange(5, 16)
    y[[1, -2]] -= 1
    y[[0, -1]] += 1
    result = stats.linregress(x, y)

    def assert_ae(x, y):
        return assert_almost_equal(x, y, decimal=14)
    assert_ae(result.slope, 1.0)
    assert_ae(result.intercept, 5.0)
    assert_ae(result.rvalue, 0.9822994862575)
    assert_ae(result.pvalue, 7.45259691e-08)
    assert_ae(result.stderr, 0.06356417261637273)
    assert_ae(result.intercept_stderr, 0.37605071654517686)