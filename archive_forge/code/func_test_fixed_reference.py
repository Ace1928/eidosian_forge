import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fixed_reference(dtype):
    img = data.text()
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    image_smooth = gaussian(img, sigma=1, preserve_range=False).astype(dtype, copy=False)
    snake = active_contour(image_smooth, init, boundary_condition='fixed', alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert snake.dtype == _supported_float_type(dtype)
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)