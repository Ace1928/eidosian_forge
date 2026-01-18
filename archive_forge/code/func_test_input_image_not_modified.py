import functools
import numpy as np
import pytest
from skimage._shared.testing import assert_
from skimage._shared.utils import _supported_float_type
from skimage.data import binary_blobs
from skimage.data import camera, chelsea
from skimage.metrics import mean_squared_error as mse
from skimage.restoration import calibrate_denoiser, denoise_wavelet
from skimage.restoration.j_invariant import denoise_invariant
from skimage.util import img_as_float, random_noise
from skimage.restoration.tests.test_denoise import xfail_without_pywt
@xfail_without_pywt
def test_input_image_not_modified():
    input_image = noisy_img.copy()
    parameter_ranges = {'sigma': np.random.random(5) / 2}
    calibrate_denoiser(input_image, _denoise_wavelet, denoise_parameters=parameter_ranges)
    assert_(np.all(noisy_img == input_image))