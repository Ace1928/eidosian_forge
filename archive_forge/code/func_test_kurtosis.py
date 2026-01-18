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
def test_kurtosis(self):
    y = stats.kurtosis(self.scalar_testcase)
    assert np.isnan(y)
    y = stats.kurtosis(self.testmathworks, 0, fisher=0, bias=1)
    assert_approx_equal(y, 2.1658856802973, 10)
    y = stats.kurtosis(self.testmathworks, fisher=0, bias=0)
    assert_approx_equal(y, 3.663542721189047, 10)
    y = stats.kurtosis(self.testcase, 0, 0)
    assert_approx_equal(y, 1.64)
    x = np.arange(10.0)
    x[9] = np.nan
    assert_equal(stats.kurtosis(x), np.nan)
    assert_almost_equal(stats.kurtosis(x, nan_policy='omit'), -1.23)
    assert_raises(ValueError, stats.kurtosis, x, nan_policy='raise')
    assert_raises(ValueError, stats.kurtosis, x, nan_policy='foobar')