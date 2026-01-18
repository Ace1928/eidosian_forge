import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_1d_ma(self):
    a = ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    desired = 45.2872868812
    check_equal_gmean(a, desired)
    a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
    desired = np.power(1 * 2 * 3, 1.0 / 3.0)
    check_equal_gmean(a, desired, rtol=1e-14)