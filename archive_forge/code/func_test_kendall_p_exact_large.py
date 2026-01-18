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
@pytest.mark.xslow
def test_kendall_p_exact_large(self):
    expectations = {(400, 38965): 0.48444283672113314, (401, 39516): 0.6636315982347484, (800, 156772): 0.4226544848312093, (801, 157849): 0.5343755341219442, (1600, 637472): 0.8420072740032354, (1601, 630304): 0.34465255088058594}
    for nc, expected in expectations.items():
        res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
        assert_almost_equal(res, expected)