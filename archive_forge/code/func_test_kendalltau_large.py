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
@pytest.mark.skipif(platform.machine() == 'ppc64le', reason='fails/crashes on ppc64le')
@pytest.mark.slow
def test_kendalltau_large(self):
    x = np.arange(2000, dtype=float)
    x = ma.masked_greater(x, 1995)
    y = np.arange(2000, dtype=float)
    y = np.concatenate((y[1000:], y[:1000]))
    assert_(np.isfinite(mstats.kendalltau(x, y)[1]))