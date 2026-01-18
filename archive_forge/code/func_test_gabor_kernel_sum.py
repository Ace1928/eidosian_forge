import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.filters._gabor import _sigma_prefactor, gabor, gabor_kernel
def test_gabor_kernel_sum():
    for sigma_x in range(1, 10, 2):
        for sigma_y in range(1, 10, 2):
            for frequency in range(0, 10, 2):
                kernel = gabor_kernel(frequency + 0.1, theta=0, sigma_x=sigma_x, sigma_y=sigma_y)
                assert_almost_equal(np.abs(kernel).sum(), 1, 2)