import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_bg_and_color_cycle():
    image = np.zeros((1, 10))
    label = np.arange(10).reshape(1, -1)
    colors = [(1, 0, 0), (0, 0, 1)]
    bg_color = (0, 0, 0)
    rgb = label2rgb(label, image=image, bg_label=0, bg_color=bg_color, colors=colors, alpha=1)
    assert_array_almost_equal(rgb[0, 0], bg_color)
    for pixel, color in zip(rgb[0, 1:], itertools.cycle(colors)):
        assert_array_almost_equal(pixel, color)