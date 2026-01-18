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
def generate_xy_sample2D(self, n, nx):
    x = np.full((n, nx), np.nan)
    y = np.full((n, nx), np.nan)
    xm = np.full((n + 5, nx), np.nan)
    ym = np.full((n + 5, nx), np.nan)
    for i in range(nx):
        x[:, i], y[:, i], dx, dy = self.generate_xy_sample(n)
    xm[0:n, :] = x[0:n]
    ym[0:n, :] = y[0:n]
    xm = np.ma.array(xm, mask=np.isnan(xm))
    ym = np.ma.array(ym, mask=np.isnan(ym))
    return (x, y, xm, ym)