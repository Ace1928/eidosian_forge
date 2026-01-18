import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_ppf_dtype(distfn, arg):
    q0 = np.asarray([0.25, 0.5, 0.75])
    q_cast = [q0.astype(tp) for tp in (np.float16, np.float32, np.float64)]
    for q in q_cast:
        for meth in [distfn.ppf, distfn.isf]:
            val = meth(q, *arg)
            npt.assert_(val.dtype == np.float64)