from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
def test_each_channel_with_asymmetric_kernel():
    mask = np.triu(np.ones(COLOR_IMAGE.shape[:2], dtype=bool))
    mask_each(COLOR_IMAGE, mask)