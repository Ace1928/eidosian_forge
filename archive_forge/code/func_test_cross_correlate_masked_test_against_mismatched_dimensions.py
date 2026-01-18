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
def test_cross_correlate_masked_test_against_mismatched_dimensions():
    """Masked normalized cross-correlation should raise an error if array
    dimensions along non-transformation axes are mismatched."""
    shape1 = (23, 1, 1)
    shape2 = (6, 2, 2)
    arr1 = np.zeros(shape1)
    arr2 = np.zeros(shape2)
    m1 = np.ones_like(arr1)
    m2 = np.ones_like(arr2)
    with pytest.raises(ValueError):
        cross_correlate_masked(arr1, arr2, m1, m2, axes=(1, 2))