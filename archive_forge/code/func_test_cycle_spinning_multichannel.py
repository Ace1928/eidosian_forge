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
@pytest.mark.parametrize('channel_axis', [-1, None])
@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_cycle_spinning_multichannel(rescale_sigma, channel_axis):
    sigma = 0.1
    rstate = np.random.default_rng(1234)
    if channel_axis is not None:
        img = astro
        valid_shifts = [1, (0, 1), (1, 0), (1, 1), (1, 1, 0)]
        valid_steps = [1, 2, (1, 2), (1, 2, 1)]
        invalid_shifts = [(1, 1, 2), (1,), (1, 1, 0, 1)]
        invalid_steps = [(1,), (1, 1, 1, 1), (0, 1), (-1, -1)]
    else:
        img = astro_gray
        valid_shifts = [1, (0, 1), (1, 0), (1, 1)]
        valid_steps = [1, 2, (1, 2)]
        invalid_shifts = [(1, 1, 2), (1,)]
        invalid_steps = [(1,), (1, 1, 1), (0, 1), (-1, -1)]
    noisy = img.copy() + 0.1 * rstate.standard_normal(img.shape)
    denoise_func = restoration.denoise_wavelet
    func_kw = dict(sigma=sigma, channel_axis=channel_axis, rescale_sigma=rescale_sigma)
    with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
        dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=0, func_kw=func_kw, channel_axis=channel_axis)
        dn = denoise_func(noisy, **func_kw)
    assert_array_equal(dn, dn_cc)
    for max_shifts in valid_shifts:
        with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
            dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=max_shifts, func_kw=func_kw, channel_axis=channel_axis)
        psnr = peak_signal_noise_ratio(img, dn)
        psnr_cc = peak_signal_noise_ratio(img, dn_cc)
        assert psnr_cc > psnr
    for shift_steps in valid_steps:
        with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
            dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=2, shift_steps=shift_steps, func_kw=func_kw, channel_axis=channel_axis)
        psnr = peak_signal_noise_ratio(img, dn)
        psnr_cc = peak_signal_noise_ratio(img, dn_cc)
        assert psnr_cc > psnr
    for max_shifts in invalid_shifts:
        with pytest.raises(ValueError):
            dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=max_shifts, func_kw=func_kw, channel_axis=channel_axis)
    for shift_steps in invalid_steps:
        with pytest.raises(ValueError):
            dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=2, shift_steps=shift_steps, func_kw=func_kw, channel_axis=channel_axis)