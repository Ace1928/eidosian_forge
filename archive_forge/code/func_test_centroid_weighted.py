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
def test_centroid_weighted():
    sample_for_spacing = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    target_centroid_wspacing = (4.0, 4.0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].centroid_weighted
    target_centroid = (5.54054054054, 9.445945945945)
    assert_array_almost_equal(centroid, target_centroid)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    assert_almost_equal((cX, cY), centroid)
    spacing = (2, 2)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))
    assert_almost_equal(centroid, 2 * np.array(target_centroid))
    centroid = regionprops(sample_for_spacing, intensity_image=sample_for_spacing, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, 2 * np.array(target_centroid_wspacing))
    spacing = (1.3, 0.7)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))
    centroid = regionprops(sample_for_spacing, intensity_image=sample_for_spacing, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, spacing * np.array(target_centroid_wspacing))