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
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_denoise_tv_chambolle_multichannel(channel_axis):
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)
    img = np.moveaxis(astro, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(img, weight=0.1, channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis, axis=channel_axis % img.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)
    astro3 = np.tile(astro[:64, :64, np.newaxis, :], [1, 1, 2, 1])
    astro3[:, :, 0, :] = 2 * astro3[:, :, 0, :]
    denoised0 = restoration.denoise_tv_chambolle(astro3[..., 0], weight=0.1)
    astro3 = np.moveaxis(astro3, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(astro3, weight=0.1, channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis, axis=channel_axis % astro3.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)