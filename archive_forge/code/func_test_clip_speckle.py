from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_clip_speckle():
    data = camera()
    data_signed = img_as_float(data) * 2.0 - 1.0
    cam_speckle = random_noise(data, mode='speckle', rng=42, clip=True)
    cam_speckle_sig = random_noise(data_signed, mode='speckle', rng=42, clip=True)
    assert cam_speckle.max() == 1.0 and cam_speckle.min() == 0.0
    assert cam_speckle_sig.max() == 1.0 and cam_speckle_sig.min() == -1.0
    cam_speckle = random_noise(data, mode='speckle', rng=42, clip=False)
    cam_speckle_sig = random_noise(data_signed, mode='speckle', rng=42, clip=False)
    assert cam_speckle.max() > 1.219 and cam_speckle.min() == 0.0
    assert cam_speckle_sig.max() > 1.219 and cam_speckle_sig.min() < -1.219