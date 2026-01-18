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
def test_axis_major_length():
    length = regionprops(SAMPLE)[0].axis_major_length
    target_length = 16.7924234999
    assert_almost_equal(length, target_length)
    length = regionprops(SAMPLE, spacing=(2, 2))[0].axis_major_length
    assert_almost_equal(length, 2 * target_length)
    from skimage.draw import ellipse
    img = np.zeros((20, 24), dtype=np.uint8)
    rr, cc = ellipse(11, 11, 7, 9, rotation=np.deg2rad(45))
    img[rr, cc] = 1
    target_length = regionprops(img, spacing=(1, 1))[0].axis_major_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[0].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[:, ::2], spacing=(1, 2))[0].axis_major_length
    assert_almost_equal(length, target_length, decimal=0)