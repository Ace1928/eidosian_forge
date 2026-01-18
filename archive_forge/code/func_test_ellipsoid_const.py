import numpy as np
import pytest
from skimage import data
from skimage.restoration._rolling_ball import rolling_ball
from skimage.restoration._rolling_ball import ellipsoid_kernel
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_ellipsoid_const(dtype):
    img = 155 * np.ones((100, 100), dtype=dtype)
    kernel = ellipsoid_kernel((25, 53), 50)
    background = rolling_ball(img, kernel=kernel)
    assert np.allclose(img - background, np.zeros_like(img))
    assert background.dtype == img.dtype