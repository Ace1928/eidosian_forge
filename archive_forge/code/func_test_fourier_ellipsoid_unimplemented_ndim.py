import numpy
from numpy import fft
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
import pytest
from scipy import ndimage
def test_fourier_ellipsoid_unimplemented_ndim(self):
    x = numpy.ones((4, 6, 8, 10), dtype=numpy.complex128)
    with pytest.raises(NotImplementedError):
        ndimage.fourier_ellipsoid(x, 3)