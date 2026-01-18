import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_meth_dtype(distfn, arg, meths):
    q0 = [0.25, 0.5, 0.75]
    x0 = distfn.ppf(q0, *arg)
    x_cast = [x0.astype(tp) for tp in (np_long, np.float16, np.float32, np.float64)]
    for x in x_cast:
        distfn._argcheck(*arg)
        x = x[(distfn.a < x) & (x < distfn.b)]
        for meth in meths:
            val = meth(x, *arg)
            npt.assert_(val.dtype == np.float64)