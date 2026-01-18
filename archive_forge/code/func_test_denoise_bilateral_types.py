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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_denoise_bilateral_types(dtype):
    img = checkerboard_gray.copy()[:50, :50]
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1).astype(dtype)
    restoration.denoise_bilateral(img, sigma_color=0.1, sigma_spatial=10, channel_axis=None)