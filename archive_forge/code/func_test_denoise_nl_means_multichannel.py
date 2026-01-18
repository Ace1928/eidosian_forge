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
@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
@pytest.mark.parametrize('channel_axis', [0, -1])
def test_denoise_nl_means_multichannel(fast_mode, dtype, channel_axis):
    img = data.binary_blobs(length=32, n_dim=3, rng=5)
    img = img[:, :24, :16].astype(dtype, copy=False)
    sigma = 0.2
    rng = np.random.default_rng(5)
    imgn = img + sigma * rng.standard_normal(img.shape)
    imgn = imgn.astype(dtype)
    denoised_ok_multichannel = restoration.denoise_nl_means(imgn.copy(), 3, 2, h=0.6 * sigma, sigma=sigma, fast_mode=fast_mode, channel_axis=None)
    imgn = np.moveaxis(imgn, -1, channel_axis)
    denoised_wrong_multichannel = restoration.denoise_nl_means(imgn.copy(), 3, 2, h=0.6 * sigma, sigma=sigma, fast_mode=fast_mode, channel_axis=channel_axis)
    denoised_wrong_multichannel = np.moveaxis(denoised_wrong_multichannel, channel_axis, -1)
    img = img.astype(denoised_wrong_multichannel.dtype)
    psnr_wrong = peak_signal_noise_ratio(img, denoised_wrong_multichannel)
    psnr_ok = peak_signal_noise_ratio(img, denoised_ok_multichannel)
    assert psnr_ok > psnr_wrong