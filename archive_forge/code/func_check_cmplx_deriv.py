import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_cmplx_deriv(distfn, arg):

    def deriv(f, x, *arg):
        x = np.asarray(x)
        h = 1e-10
        return (f(x + h * 1j, *arg) / h).imag
    x0 = distfn.ppf([0.25, 0.51, 0.75], *arg)
    x_cast = [x0.astype(tp) for tp in (np_long, np.float16, np.float32, np.float64)]
    for x in x_cast:
        distfn._argcheck(*arg)
        x = x[(distfn.a < x) & (x < distfn.b)]
        pdf, cdf, sf = (distfn.pdf(x, *arg), distfn.cdf(x, *arg), distfn.sf(x, *arg))
        assert_allclose(deriv(distfn.cdf, x, *arg), pdf, rtol=1e-05)
        assert_allclose(deriv(distfn.logcdf, x, *arg), pdf / cdf, rtol=1e-05)
        assert_allclose(deriv(distfn.sf, x, *arg), -pdf, rtol=1e-05)
        assert_allclose(deriv(distfn.logsf, x, *arg), -pdf / sf, rtol=1e-05)
        assert_allclose(deriv(distfn.logpdf, x, *arg), deriv(distfn.pdf, x, *arg) / distfn.pdf(x, *arg), rtol=1e-05)