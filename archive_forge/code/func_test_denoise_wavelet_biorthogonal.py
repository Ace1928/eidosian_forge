import functools
import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_warns
from skimage import color, data, img_as_float, restoration
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration._denoise import _wavelet_threshold
@xfail_without_pywt
@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_denoise_wavelet_biorthogonal(rescale_sigma):
    """Biorthogonal wavelets should raise a warning during thresholding."""
    img = astro_gray
    assert_warns(UserWarning, restoration.denoise_wavelet, img, wavelet='bior2.2', channel_axis=None, rescale_sigma=rescale_sigma)