import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, suppress_warnings
from scipy.special._ufuncs import _sinpi as sinpi
from scipy.special._ufuncs import _cospi as cospi
@pytest.mark.skip('Temporary skip while gh-19526 is being resolved')
def test_intermediate_overlow():
    sinpi_pts = [complex(1 + 1e-14, 227), complex(1e-35, 250), complex(1e-301, 445)]
    sinpi_std = [complex(-8.113438309924894e+295, -np.inf), complex(1.9507801934611995e+306, np.inf), complex(2.205958493464539e+306, np.inf)]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
        for p, std in zip(sinpi_pts, sinpi_std):
            res = sinpi(p)
            assert_allclose(res.real, std.real)
            assert_allclose(res.imag, std.imag)
    p = complex(0.5 + 1e-14, 227)
    std = complex(-8.113438309924894e+295, -np.inf)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
        res = cospi(p)
        assert_allclose(res.real, std.real)
        assert_allclose(res.imag, std.imag)