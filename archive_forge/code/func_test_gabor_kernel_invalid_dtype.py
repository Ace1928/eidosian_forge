import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
@pytest.mark.parametrize('dtype', [np.uint8, np.float32])
def test_gabor_kernel_invalid_dtype(dtype):
    with pytest.raises(ValueError):
        kernel = gabor_kernel(1, bandwidth=1, dtype=dtype)
        assert kernel.dtype == dtype