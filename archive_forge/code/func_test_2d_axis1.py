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
def test_2d_axis1(self):
    a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
    desired = np.array([22.13363839, 64.02171746, 104.40086817])
    check_equal_gmean(a, desired, axis=1)
    a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    v = power(1 * 2 * 3 * 4, 1.0 / 4.0)
    desired = array([v, v, v])
    check_equal_gmean(a, desired, axis=1, rtol=1e-14)