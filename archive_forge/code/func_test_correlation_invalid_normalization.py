import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import fourier_shift
import scipy.fft as fft
from skimage import img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_stacklevel
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, binary_blobs, eagle
from skimage.registration._phase_cross_correlation import (
@pytest.mark.parametrize('normalization', ['nonexisting'])
def test_correlation_invalid_normalization(normalization):
    reference_image = fft.fftn(camera())
    shift = (-7, 12)
    shifted_image = fourier_shift(reference_image, shift)
    with pytest.raises(ValueError):
        phase_cross_correlation(reference_image, shifted_image, space='fourier', normalization=normalization)