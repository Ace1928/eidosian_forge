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
def test_nd_input(self):
    x = np.array((-2, -1, 0, 1, 2, 3) * 4) ** 2
    x_2d = np.vstack([x] * 2).T
    for func in [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]:
        res_1d = func(x)
        res_2d = func(x_2d)
        assert_allclose(res_2d[0], [res_1d[0]] * 2)
        assert_allclose(res_2d[1], [res_1d[1]] * 2)