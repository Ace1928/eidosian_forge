import numpy.testing as npt
from scipy.optimize.cython_optimize import _zeros
def test_brenth():
    npt.assert_allclose(EXPECTED, list(_zeros.loop_example('brenth', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)), rtol=RTOL, atol=XTOL)