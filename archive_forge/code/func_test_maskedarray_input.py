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
def test_maskedarray_input(self):
    x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
    xm = np.ma.array(np.r_[np.inf, x, 10], mask=np.r_[True, [False] * x.size, True])
    assert_allclose(mstats.normaltest(xm), stats.normaltest(x))
    assert_allclose(mstats.skewtest(xm), stats.skewtest(x))
    assert_allclose(mstats.kurtosistest(xm), stats.kurtosistest(x))