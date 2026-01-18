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
@pytest.mark.parametrize('axis, expected', [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
def test_center_mean_with_nan(self, axis, expected):
    x = np.array([[1, 2, 4, 9, np.nan], [0, 1, 1, 1, 12], [-10, -10, -10, 20, 20]])
    mad = stats.median_abs_deviation(x, center=np.mean, nan_policy='omit', axis=axis)
    assert_allclose(mad, expected, rtol=1e-15, atol=1e-15)