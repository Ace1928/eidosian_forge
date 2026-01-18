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
@pytest.mark.parametrize('fast_mode', [False, True])
def test_denoise_nl_means_2d_dtype(fast_mode):
    img = np.zeros((40, 40), dtype=int)
    img_f32 = img.astype('float32')
    img_f64 = img.astype('float64')
    assert restoration.denoise_nl_means(img, fast_mode=fast_mode).dtype == 'float64'
    assert restoration.denoise_nl_means(img_f32, fast_mode=fast_mode).dtype == img_f32.dtype
    assert restoration.denoise_nl_means(img_f64, fast_mode=fast_mode).dtype == img_f64.dtype