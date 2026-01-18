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
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize('expect, moment', [(0, 1), (1, 0)])
def test_constant_moments(self, dtype, expect, moment):
    x = np.random.rand(5).astype(dtype)
    y = stats.moment(x, moment=moment)
    self._assert_equal(y, expect, dtype=dtype)
    y = stats.moment(np.broadcast_to(x, (6, 5)), axis=0, moment=moment)
    self._assert_equal(y, expect, shape=(5,), dtype=dtype)
    y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=2, moment=moment)
    self._assert_equal(y, expect, shape=(1, 2, 4, 5), dtype=dtype)
    y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=None, moment=moment)
    self._assert_equal(y, expect, shape=(), dtype=dtype)