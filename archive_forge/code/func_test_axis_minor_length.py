import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_axis_minor_length():
    length = regionprops(SAMPLE)[0].axis_minor_length
    target_length = 9.739302807263
    assert_almost_equal(length, target_length)
    length = regionprops(SAMPLE, spacing=(1.5, 1.5))[0].axis_minor_length
    assert_almost_equal(length, 1.5 * target_length)
    from skimage.draw import ellipse
    img = np.zeros((10, 12), dtype=np.uint8)
    rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    img[rr, cc] = 1
    target_length = regionprops(img, spacing=(1, 1))[0].axis_minor_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[0].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[::2], spacing=(2, 1))[0].axis_minor_length
    assert_almost_equal(length, target_length, decimal=1)