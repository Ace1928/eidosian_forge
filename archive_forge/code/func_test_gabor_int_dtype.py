import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.intp])
def test_gabor_int_dtype(dtype):
    image = np.full((16, 16), 128, dtype=dtype)
    y = gabor(image, 0.3)
    assert all((arr.dtype == dtype for arr in y))