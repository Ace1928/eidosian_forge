import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_moment(distfn, arg, m, v, msg):
    m1 = distfn.moment(1, *arg)
    m2 = distfn.moment(2, *arg)
    if not np.isinf(m):
        npt.assert_almost_equal(m1, m, decimal=10, err_msg=msg + ' - 1st moment')
    else:
        npt.assert_(np.isinf(m1), msg + ' - 1st moment -infinite, m1=%s' % str(m1))
    if not np.isinf(v):
        npt.assert_almost_equal(m2 - m1 * m1, v, decimal=10, err_msg=msg + ' - 2ndt moment')
    else:
        npt.assert_(np.isinf(m2), msg + f' - 2nd moment -infinite, m2={m2!r}')