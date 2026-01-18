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
def test_equivalent_diameter_area():
    diameter = regionprops(SAMPLE)[0].equivalent_diameter_area
    assert_almost_equal(diameter, 9.57461472963)
    spacing = (1, 3)
    diameter = regionprops(SAMPLE, spacing=spacing)[0].equivalent_diameter_area
    equivalent_area = np.pi * (diameter / 2.0) ** 2
    assert_almost_equal(equivalent_area, SAMPLE.sum() * np.prod(spacing))