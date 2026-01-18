from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
def test_each_channel():
    filtered = edges_each(COLOR_IMAGE)
    for i, channel in enumerate(np.rollaxis(filtered, axis=-1)):
        expected = img_as_float(filters.sobel(COLOR_IMAGE[:, :, i]))
        assert_allclose(channel, expected)