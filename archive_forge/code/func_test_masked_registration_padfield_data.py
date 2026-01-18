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
def test_masked_registration_padfield_data():
    """Masked translation registration should behave like in the original
    publication"""
    shifts = [(75, 75), (-130, 130), (130, 130)]
    for xi, yi in shifts:
        fixed_image = imread(fetch(f'registration/tests/data/OriginalX{xi}Y{yi}.png'))
        moving_image = imread(fetch(f'registration/tests/data/TransformedX{xi}Y{yi}.png'))
        fixed_mask = fixed_image != 0
        moving_mask = moving_image != 0
        shift_y, shift_x = masked_register_translation(fixed_image, moving_image, reference_mask=fixed_mask, moving_mask=moving_mask, overlap_ratio=0.1)
        assert_equal((shift_x, shift_y), (-xi, yi))