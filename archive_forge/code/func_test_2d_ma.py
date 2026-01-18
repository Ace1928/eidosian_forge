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
def test_2d_ma(self):
    a = ma.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], mask=[[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
    desired = np.array([1, 2, 3, 4])
    check_equal_gmean(a, desired, axis=0, rtol=1e-14)
    desired = ma.array([np.power(1 * 2 * 3 * 4, 1.0 / 4.0), np.power(2 * 3, 1.0 / 2.0), np.power(1 * 4, 1.0 / 2.0)])
    check_equal_gmean(a, desired, axis=-1, rtol=1e-14)
    a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
    desired = 52.8885199
    check_equal_gmean(np.ma.array(a), desired)