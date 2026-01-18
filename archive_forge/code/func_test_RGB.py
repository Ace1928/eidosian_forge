import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_RGB(dtype):
    img = gaussian(data.text(), sigma=1, preserve_range=False)
    imgR = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgG = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgRGB = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgR[:, :, 0] = img
    imgG[:, :, 1] = img
    imgRGB[:, :, :] = img[:, :, None]
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    snake = active_contour(imgR, init, boundary_condition='fixed', alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    float_dtype = _supported_float_type(dtype)
    assert snake.dtype == float_dtype
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgG, init, boundary_condition='fixed', alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert snake.dtype == float_dtype
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgRGB, init, boundary_condition='fixed', alpha=0.1, beta=1.0, w_line=-5 / 3.0, w_edge=0, gamma=0.1)
    assert snake.dtype == float_dtype
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)