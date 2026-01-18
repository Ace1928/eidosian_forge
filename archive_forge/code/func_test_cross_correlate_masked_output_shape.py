import numpy as np
import pytest
from numpy.testing import (
from scipy.ndimage import fourier_shift, shift as real_shift
import scipy.fft as fft
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, brain
from skimage.io import imread
from skimage.registration._masked_phase_cross_correlation import (
from skimage.registration import phase_cross_correlation
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_cross_correlate_masked_output_shape(dtype):
    """Masked normalized cross-correlation should return a shape
    of N + M + 1 for each transform axis."""
    shape1 = (15, 4, 5)
    shape2 = (6, 12, 7)
    expected_full_shape = tuple(np.array(shape1) + np.array(shape2) - 1)
    expected_same_shape = shape1
    arr1 = np.zeros(shape1, dtype=dtype)
    arr2 = np.zeros(shape2, dtype=dtype)
    m1 = np.ones_like(arr1)
    m2 = np.ones_like(arr2)
    float_dtype = _supported_float_type(dtype)
    full_xcorr = cross_correlate_masked(arr1, arr2, m1, m2, axes=(0, 1, 2), mode='full')
    assert_equal(full_xcorr.shape, expected_full_shape)
    assert full_xcorr.dtype == float_dtype
    same_xcorr = cross_correlate_masked(arr1, arr2, m1, m2, axes=(0, 1, 2), mode='same')
    assert_equal(same_xcorr.shape, expected_same_shape)
    assert same_xcorr.dtype == float_dtype