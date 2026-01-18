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
def test_feret_diameter_max():
    comparator_result = 18
    test_result = regionprops(SAMPLE)[0].feret_diameter_max
    assert np.abs(test_result - comparator_result) < 1
    comparator_result_spacing = 10
    test_result_spacing = regionprops(SAMPLE, spacing=[1, 0.1])[0].feret_diameter_max
    assert np.abs(test_result_spacing - comparator_result_spacing) < 1
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    feret_diameter_max = regionprops(img)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - 16 * np.sqrt(2)) < 1
    assert np.abs(feret_diameter_max - np.sqrt(16 ** 2 + (16 - 1) ** 2)) < 1e-06