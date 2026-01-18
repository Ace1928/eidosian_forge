import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_avg(channel_axis):
    label_field = np.array([[1, 1, 1, 2], [1, 2, 2, 2], [3, 3, 4, 4]], dtype=np.uint8)
    r = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    image = np.dstack((r, g, b))
    rout = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    gout = np.array([[0.25, 0.25, 0.25, 0.75], [0.25, 0.75, 0.75, 0.75], [0.0, 0.0, 0.0, 0.0]])
    bout = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    expected_out = np.dstack((rout, gout, bout))
    _image = np.moveaxis(image, source=-1, destination=channel_axis)
    out = label2rgb(label_field, _image, kind='avg', bg_label=-1, channel_axis=channel_axis)
    out = np.moveaxis(out, source=channel_axis, destination=-1)
    assert_array_equal(out, expected_out)
    out_bg = label2rgb(label_field, _image, bg_label=2, bg_color=(0, 0, 0), kind='avg', channel_axis=channel_axis)
    out_bg = np.moveaxis(out_bg, source=channel_axis, destination=-1)
    expected_out_bg = expected_out.copy()
    expected_out_bg[label_field == 2] = 0
    assert_array_equal(out_bg, expected_out_bg)
    out_bg = label2rgb(label_field, _image, bg_label=2, kind='avg', channel_axis=channel_axis)
    out_bg = np.moveaxis(out_bg, source=channel_axis, destination=-1)
    assert_array_equal(out_bg, expected_out_bg)