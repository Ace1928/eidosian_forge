from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
def test_hsv_value_with_non_float_output():
    filtered = edges_hsv_uint(COLOR_IMAGE)
    filtered_value = color.rgb2hsv(filtered)[:, :, 2]
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(filtered_value, filters.sobel(value), rtol=1e-05, atol=1e-05)