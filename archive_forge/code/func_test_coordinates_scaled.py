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
@pytest.mark.parametrize('spacing', [None, 1, 2, (1, 1), (1, 0.5)])
def test_coordinates_scaled(spacing):
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    if spacing is None:
        desired_coords = coords
    else:
        desired_coords = coords * np.array(spacing)
    assert_array_equal(prop_coords, desired_coords)