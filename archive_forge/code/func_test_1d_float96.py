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
@pytest.mark.skipif(not hasattr(np, 'float96'), reason='cannot find float96 so skipping')
def test_1d_float96(self):
    a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
    desired_dt = np.asarray(3.0 / (1.0 / 1 + 1.0 / 2 + 1.0 / 3), dtype=np.float96)
    check_equal_hmean(a, desired_dt, dtype=np.float96)