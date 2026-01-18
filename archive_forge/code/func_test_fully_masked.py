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
def test_fully_masked(self):
    np.random.seed(1234567)
    outcome = ma.masked_array(np.random.randn(3), mask=[1, 1, 1])
    expected = (np.nan, np.nan)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in absolute')
        for pair in [((np.nan, np.nan), 0.0), (outcome, 0.0)]:
            t, p = mstats.ttest_1samp(*pair)
            assert_array_equal(p, expected)
            assert_array_equal(t, expected)