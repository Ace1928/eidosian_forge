import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_normalization(distfn, args, distname):
    norm_moment = distfn.moment(0, *args)
    npt.assert_allclose(norm_moment, 1.0)
    if distname == 'rv_histogram_instance':
        atol, rtol = (1e-05, 0)
    else:
        atol, rtol = (1e-07, 1e-07)
    normalization_expect = distfn.expect(lambda x: 1, args=args)
    npt.assert_allclose(normalization_expect, 1.0, atol=atol, rtol=rtol, err_msg=distname, verbose=True)
    _a, _b = distfn.support(*args)
    normalization_cdf = distfn.cdf(_b, *args)
    npt.assert_allclose(normalization_cdf, 1.0)