import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_label_consistency():
    """Assert that the same labels map to the same colors."""
    label_1 = np.arange(5).reshape(1, -1)
    label_2 = np.array([0, 1])
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    rgb_1 = label2rgb(label_1, colors=colors, bg_label=-1)
    rgb_2 = label2rgb(label_2, colors=colors, bg_label=-1)
    for label_id in label_2.flat:
        assert_array_almost_equal(rgb_1[label_1 == label_id], rgb_2[label_2 == label_id])