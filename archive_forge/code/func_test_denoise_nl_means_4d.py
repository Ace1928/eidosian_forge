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
def test_denoise_nl_means_4d():
    rng = np.random.default_rng(5)
    img = np.zeros((10, 10, 8, 5))
    img[2:-2, 2:-2, 2:-2, :2] = 0.5
    img[2:-2, 2:-2, 2:-2, 2:] = 1.0
    sigma = 0.3
    imgn = img + sigma * rng.standard_normal(img.shape)
    nlmeans_kwargs = dict(patch_size=3, patch_distance=2, h=0.3 * sigma, sigma=sigma, fast_mode=True)
    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.0)
    denoised_3d = np.zeros_like(imgn)
    for ch in range(img.shape[-1]):
        denoised_3d[..., ch] = restoration.denoise_nl_means(imgn[..., ch], channel_axis=None, **nlmeans_kwargs)
    psnr_3d = peak_signal_noise_ratio(img, denoised_3d, data_range=1.0)
    assert psnr_3d > psnr_noisy
    denoised_4d = restoration.denoise_nl_means(imgn, channel_axis=None, **nlmeans_kwargs)
    psnr_4d = peak_signal_noise_ratio(img, denoised_4d, data_range=1.0)
    assert psnr_4d > psnr_3d
    denoised_3dmc = restoration.denoise_nl_means(imgn, channel_axis=-1, **nlmeans_kwargs)
    psnr_3dmc = peak_signal_noise_ratio(img, denoised_3dmc, data_range=1.0)
    assert psnr_3dmc > psnr_3d