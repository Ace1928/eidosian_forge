import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_PSNR_float(dtype):
    p_uint8 = peak_signal_noise_ratio(cam, cam_noisy)
    camf = (cam / 255.0).astype(dtype, copy=False)
    camf_noisy = (cam_noisy / 255.0).astype(dtype, copy=False)
    p_float64 = peak_signal_noise_ratio(camf, camf_noisy, data_range=1)
    assert p_float64.dtype == np.float64
    decimal = 3 if dtype == np.float16 else 5
    assert_almost_equal(p_uint8, p_float64, decimal=decimal)
    p_mixed = peak_signal_noise_ratio(cam / 255.0, np.float32(cam_noisy / 255.0), data_range=1)
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)
    with expected_warnings(['Inputs have mismatched dtype']):
        p_mixed = peak_signal_noise_ratio(cam / 255.0, np.float32(cam_noisy / 255.0))
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)
    with expected_warnings(['Inputs have mismatched dtype']):
        p_mixed = peak_signal_noise_ratio(cam / 255.0, np.float32(cam_noisy / 255.0))
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)