import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_avg_with_2d_image():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    assert_no_warnings(label2rgb, labels, image=img, bg_label=0, kind='avg')