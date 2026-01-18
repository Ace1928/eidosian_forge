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
def test_describe_ddof(self):
    x = np.vstack((np.ones((3, 4)), np.full((2, 4), 2)))
    nc, mmc = (5, ([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]))
    mc = np.array([1.4, 1.4, 1.4, 1.4])
    vc = np.array([0.24, 0.24, 0.24, 0.24])
    skc = [0.4082482904638636] * 4
    kurtc = [-1.833333333333333] * 4
    n, mm, m, v, sk, kurt = stats.describe(x, ddof=0)
    assert_equal(n, nc)
    assert_allclose(mm, mmc, rtol=1e-15)
    assert_allclose(m, mc, rtol=1e-15)
    assert_allclose(v, vc, rtol=1e-15)
    assert_array_almost_equal(sk, skc, decimal=13)
    assert_array_almost_equal(kurt, kurtc, decimal=13)