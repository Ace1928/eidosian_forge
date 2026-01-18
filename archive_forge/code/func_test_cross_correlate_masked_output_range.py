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
def test_cross_correlate_masked_output_range():
    """Masked normalized cross-correlation should return between 1 and -1."""
    np.random.seed(23)
    shape1 = (15, 4, 5)
    shape2 = (15, 12, 7)
    arr1 = 10 * np.random.random(shape1) - 5
    arr2 = 10 * np.random.random(shape2) - 5
    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)
    xcorr = cross_correlate_masked(arr1, arr2, m1, m2, axes=(1, 2))
    eps = np.finfo(float).eps
    assert_array_less(xcorr, 1 + eps)
    assert_array_less(-xcorr, 1 + eps)