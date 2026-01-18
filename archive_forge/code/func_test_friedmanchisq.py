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
def test_friedmanchisq(self):
    args = ([9.0, 9.5, 5.0, 7.5, 9.5, 7.5, 8.0, 7.0, 8.5, 6.0], [7.0, 6.5, 7.0, 7.5, 5.0, 8.0, 6.0, 6.5, 7.0, 7.0], [6.0, 8.0, 4.0, 6.0, 7.0, 6.5, 6.0, 4.0, 6.5, 3.0])
    result = mstats.friedmanchisquare(*args)
    assert_almost_equal(result[0], 10.4737, 4)
    assert_almost_equal(result[1], 0.005317, 6)
    x = [[nan, nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1], [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3], [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan], [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
    x = ma.fix_invalid(x)
    result = mstats.friedmanchisquare(*x)
    assert_almost_equal(result[0], 2.0156, 4)
    assert_almost_equal(result[1], 0.5692, 4)
    attributes = ('statistic', 'pvalue')
    check_named_results(result, attributes, ma=True)