import numpy
from numpy import fft
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
import pytest
from scipy import ndimage
@pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
@pytest.mark.parametrize('dtype, dec', [(numpy.complex64, 5), (numpy.complex128, 14)])
def test_fourier_ellipsoid_complex01(self, shape, dtype, dec):
    a = numpy.zeros(shape, dtype)
    a[0, 0] = 1.0
    a = fft.fft(a, shape[0], 0)
    a = fft.fft(a, shape[1], 1)
    a = ndimage.fourier_ellipsoid(a, [5.0, 2.5], -1, 0)
    a = fft.ifft(a, shape[1], 1)
    a = fft.ifft(a, shape[0], 0)
    assert_almost_equal(ndimage.sum(a.real), 1.0, decimal=dec)