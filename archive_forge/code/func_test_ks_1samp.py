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
def test_ks_1samp(self):
    """Checks that mstats.ks_1samp and stats.ks_1samp agree on masked arrays."""
    for mode in ['auto', 'exact', 'asymp']:
        with suppress_warnings():
            for alternative in ['less', 'greater', 'two-sided']:
                for n in self.get_n():
                    x, y, xm, ym = self.generate_xy_sample(n)
                    res1 = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
                    res2 = stats.mstats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                    assert_equal(np.asarray(res1), np.asarray(res2))
                    res3 = stats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                    assert_equal(np.asarray(res1), np.asarray(res3))