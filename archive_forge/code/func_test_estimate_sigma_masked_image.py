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
def test_estimate_sigma_masked_image():
    rstate = np.random.default_rng(1234)
    img = np.zeros((128, 128))
    center_roi = (slice(32, 96), slice(32, 96))
    img[center_roi] = 0.8
    sigma = 0.1
    img[center_roi] = sigma * rstate.standard_normal(img[center_roi].shape)
    sigma_est = restoration.estimate_sigma(img, channel_axis=None)
    assert_array_almost_equal(sigma, sigma_est, decimal=1)