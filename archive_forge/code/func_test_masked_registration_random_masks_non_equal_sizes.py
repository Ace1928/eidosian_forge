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
def test_masked_registration_random_masks_non_equal_sizes():
    """masked_register_translation should be able to register
    translations between images that are not the same size even
    with random masks."""
    np.random.seed(23)
    reference_image = camera()
    shift = (-7, 12)
    shifted = np.real(fft.ifft2(fourier_shift(fft.fft2(reference_image), shift)))
    shifted = shifted[64:-64, 64:-64]
    ref_mask = np.random.choice([True, False], reference_image.shape, p=[3 / 4, 1 / 4])
    shifted_mask = np.random.choice([True, False], shifted.shape, p=[3 / 4, 1 / 4])
    measured_shift = masked_register_translation(reference_image, shifted, reference_mask=np.ones_like(ref_mask), moving_mask=np.ones_like(shifted_mask))
    assert_equal(measured_shift, -np.array(shift))