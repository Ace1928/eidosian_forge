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
def test_coordinates():
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)
    prop_coords = regionprops(sample, spacing=(0.5, 1.2))[0].coords
    assert_array_equal(prop_coords, coords)