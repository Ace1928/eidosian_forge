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
def test__broadcast_concatenate():
    np.random.seed(0)
    a = np.random.rand(5, 4, 4, 3, 1, 6)
    b = np.random.rand(4, 1, 8, 2, 6)
    c = _broadcast_concatenate((a, b), axis=-3)
    a = np.tile(a, (1, 1, 1, 1, 2, 1))
    b = np.tile(b[None, ...], (5, 1, 4, 1, 1, 1))
    for index in product(*(range(i) for i in c.shape)):
        i, j, k, l, m, n = index
        if l < a.shape[-3]:
            assert a[i, j, k, l, m, n] == c[i, j, k, l, m, n]
        else:
            assert b[i, j, k, l - a.shape[-3], m, n] == c[i, j, k, l, m, n]