from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_clip_poisson():
    data = camera()
    data_signed = img_as_float(data) * 2.0 - 1.0
    cam_poisson = random_noise(data, mode='poisson', rng=42, clip=True)
    cam_poisson2 = random_noise(data_signed, mode='poisson', rng=42, clip=True)
    assert cam_poisson.max() == 1.0 and cam_poisson.min() == 0.0
    assert cam_poisson2.max() == 1.0 and cam_poisson2.min() == -1.0
    cam_poisson = random_noise(data, mode='poisson', rng=42, clip=False)
    cam_poisson2 = random_noise(data_signed, mode='poisson', rng=42, clip=False)
    assert cam_poisson.max() > 1.15 and cam_poisson.min() == 0.0
    assert cam_poisson2.max() > 1.3 and cam_poisson2.min() == -1.0