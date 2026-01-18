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
def test_moments_weighted_normalized_spacing():
    spacing = (3, 3)
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted_normalized
    np.array([[np.nan, np.nan, 0.230146783, -0.0162529732], [np.nan, -0.0160405109, 0.0457932622, -0.0104598869], [0.0873590903, -0.0031421072, 0.0165315478, -0.0028544152], [-0.0161217406, -0.0031376984, 0.0043903193, -0.0011057191]])
    assert_almost_equal(wnu[0, 2], 0.230146783)
    assert_almost_equal(wnu[0, 3], -0.0162529732)
    assert_almost_equal(wnu[1, 1], -0.0160405109)
    assert_almost_equal(wnu[1, 2], 0.0457932622)
    assert_almost_equal(wnu[1, 3], -0.0104598869)
    assert_almost_equal(wnu[2, 0], 0.0873590903)
    assert_almost_equal(wnu[2, 1], -0.0031421072)
    assert_almost_equal(wnu[2, 2], 0.0165315478)
    assert_almost_equal(wnu[2, 3], -0.0028544152)
    assert_almost_equal(wnu[3, 0], -0.0161217406)
    assert_almost_equal(wnu[3, 1], -0.0031376984)
    assert_almost_equal(wnu[3, 2], 0.0043903193)
    assert_almost_equal(wnu[3, 3], -0.0011057191)