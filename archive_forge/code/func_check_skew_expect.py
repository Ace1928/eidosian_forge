import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_skew_expect(distfn, arg, m, v, s, msg):
    if np.isfinite(s):
        m3e = distfn.expect(lambda x: np.power(x - m, 3), arg)
        npt.assert_almost_equal(m3e, s * np.power(v, 1.5), decimal=5, err_msg=msg + ' - skew')
    else:
        npt.assert_(np.isnan(s))