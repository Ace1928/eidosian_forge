from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_clip_gaussian():
    data = camera()
    data_signed = img_as_float(data) * 2.0 - 1.0
    cam_gauss = random_noise(data, mode='gaussian', rng=42, clip=True)
    cam_gauss2 = random_noise(data_signed, mode='gaussian', rng=42, clip=True)
    assert cam_gauss.max() == 1.0 and cam_gauss.min() == 0.0
    assert cam_gauss2.max() == 1.0 and cam_gauss2.min() == -1.0
    cam_gauss = random_noise(data, mode='gaussian', rng=42, clip=False)
    cam_gauss2 = random_noise(data_signed, mode='gaussian', rng=42, clip=False)
    assert cam_gauss.max() > 1.22 and cam_gauss.min() < -0.35
    assert cam_gauss2.max() > 1.219 and cam_gauss2.min() < -1.219