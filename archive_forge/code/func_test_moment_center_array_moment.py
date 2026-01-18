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
@pytest.mark.parametrize('size', [10, (10, 2)])
@pytest.mark.parametrize('c', (None, 0, 1))
def test_moment_center_array_moment(self, size, c):
    rng = np.random.default_rng(1706828300224046506)
    x = rng.random(size=size)
    m = [0, 1, 2, 3]
    res = stats.moment(x, m, center=c)
    ref = [stats.moment(x, i, center=c) for i in m]
    assert_equal(res, ref)