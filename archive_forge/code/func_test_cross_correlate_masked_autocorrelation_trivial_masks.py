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
def test_cross_correlate_masked_autocorrelation_trivial_masks():
    """Masked normalized cross-correlation between identical arrays
    should reduce to an autocorrelation even with random masks."""
    np.random.seed(23)
    arr1 = camera()
    m1 = np.random.choice([True, False], arr1.shape, p=[3 / 4, 1 / 4])
    m2 = np.random.choice([True, False], arr1.shape, p=[3 / 4, 1 / 4])
    xcorr = cross_correlate_masked(arr1, arr1, m1, m2, axes=(0, 1), mode='same', overlap_ratio=0).real
    max_index = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    assert_almost_equal(xcorr.max(), 1, decimal=5)
    assert_array_equal(max_index, np.array(arr1.shape) / 2)