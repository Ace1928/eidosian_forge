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
def test_gmean(self):
    for n in self.get_n():
        x, y, xm, ym = self.generate_xy_sample(n)
        r = stats.gmean(abs(x))
        rm = stats.mstats.gmean(abs(xm))
        assert_allclose(r, rm, rtol=1e-13)
        r = stats.gmean(abs(y))
        rm = stats.mstats.gmean(abs(ym))
        assert_allclose(r, rm, rtol=1e-13)