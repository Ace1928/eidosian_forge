import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_leave_labels_alone():
    labels = np.array([-1, 0, 1])
    labels_saved = labels.copy()
    label2rgb(labels, bg_label=-1)
    label2rgb(labels, bg_label=1)
    assert_array_equal(labels, labels_saved)