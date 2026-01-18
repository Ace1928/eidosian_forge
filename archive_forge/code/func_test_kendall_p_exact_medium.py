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
def test_kendall_p_exact_medium(self):
    expectations = {(100, 2393): 0.6282261528795604, (101, 2436): 0.604395257735136, (170, 0): 2.755801935583541e-307, (171, 0): 0.0, (171, 1): 2.755801935583541e-307, (172, 1): 0.0, (200, 9797): 0.7475398374592968, (201, 9656): 0.40959218958120364}
    for nc, expected in expectations.items():
        res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
        assert_almost_equal(res, expected)