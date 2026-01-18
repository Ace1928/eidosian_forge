import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
def test_gabor_kernel_bandwidth():
    kernel = gabor_kernel(1, bandwidth=1)
    assert_equal(kernel.shape, (5, 5))
    kernel = gabor_kernel(1, bandwidth=0.5)
    assert_equal(kernel.shape, (9, 9))
    kernel = gabor_kernel(0.5, bandwidth=1)
    assert_equal(kernel.shape, (9, 9))