import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_edge_support(distfn, args):
    x = distfn.support(*args)
    if isinstance(distfn, stats.rv_discrete):
        x = (x[0] - 1, x[1])
    npt.assert_equal(distfn.cdf(x, *args), [0.0, 1.0])
    npt.assert_equal(distfn.sf(x, *args), [1.0, 0.0])
    if distfn.name not in ('skellam', 'dlaplace'):
        npt.assert_equal(distfn.logcdf(x, *args), [-np.inf, 0.0])
        npt.assert_equal(distfn.logsf(x, *args), [0.0, -np.inf])
    npt.assert_equal(distfn.ppf([0.0, 1.0], *args), x)
    npt.assert_equal(distfn.isf([0.0, 1.0], *args), x[::-1])
    npt.assert_(np.isnan(distfn.isf([-1, 2], *args)).all())
    npt.assert_(np.isnan(distfn.ppf([-1, 2], *args)).all())