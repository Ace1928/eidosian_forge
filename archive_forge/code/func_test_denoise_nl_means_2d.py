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
def test_denoise_nl_means_2d(fast_mode):
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.0
    sigma = 0.3
    img += sigma * np.random.standard_normal(img.shape)
    img_f32 = img.astype('float32')
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(img, 7, 5, 0.2, fast_mode=fast_mode, channel_axis=None, sigma=s)
        assert img.std() > denoised.std()
        denoised_f32 = restoration.denoise_nl_means(img_f32, 7, 5, 0.2, fast_mode=fast_mode, channel_axis=None, sigma=s)
        assert img.std() > denoised_f32.std()
        assert np.allclose(denoised_f32, denoised, atol=0.01)