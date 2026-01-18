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
def generate_xy_sample(self, n):
    np.random.seed(1234567)
    x = np.random.randn(n)
    y = x + np.random.randn(n)
    xm = np.full(len(x) + 5, 1e+16)
    ym = np.full(len(y) + 5, 1e+16)
    xm[0:len(x)] = x
    ym[0:len(y)] = y
    mask = xm > 9000000000000000.0
    xm = np.ma.array(xm, mask=mask)
    ym = np.ma.array(ym, mask=mask)
    return (x, y, xm, ym)