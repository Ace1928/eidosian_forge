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
def test_denoise_tv_chambolle_weighting():
    rstate = np.random.default_rng(1234)
    img2d = astro_gray.copy()
    img2d += 0.15 * rstate.standard_normal(img2d.shape)
    img2d = np.clip(img2d, 0, 1)
    ssim_noisy = structural_similarity(astro_gray, img2d, data_range=1.0)
    img4d = np.tile(img2d[..., None, None], (1, 1, 2, 2))
    w = 0.2
    denoised_2d = restoration.denoise_tv_chambolle(img2d, weight=w)
    denoised_4d = restoration.denoise_tv_chambolle(img4d, weight=w)
    assert denoised_2d.dtype == np.float64
    assert denoised_4d.dtype == np.float64
    ssim_2d = structural_similarity(denoised_2d, astro_gray, data_range=1.0)
    ssim = structural_similarity(denoised_2d, denoised_4d[:, :, 0, 0], data_range=1.0)
    assert ssim > 0.98
    assert ssim_2d > ssim_noisy