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
@pytest.mark.parametrize('axis', np.arange(-3, 3))
def test_mode_shape_gh_9955(self, axis, dtype=np.float64):
    rng = np.random.default_rng(984213899)
    a = rng.uniform(size=(3, 4, 5)).astype(dtype)
    res = stats.mode(a, axis=axis, keepdims=False)
    reference_shape = list(a.shape)
    reference_shape.pop(axis)
    np.testing.assert_array_equal(res.mode.shape, reference_shape)
    np.testing.assert_array_equal(res.count.shape, reference_shape)