import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@run_in_parallel()
def test_corner_orientations_astronaut():
    img = rgb2gray(data.astronaut())
    corners = corner_peaks(corner_fast(img, 11, 0.35), min_distance=10, threshold_abs=0, threshold_rel=0.1)
    expected = np.array([-0.440598471, -1.46554357, 2.39291733, -1.63869275, 1.45931342, -1.64397304, -1.76069982, 1.09650167, -1.65449964, 1.19134149, 0.0546905279, 2.17103132, 0.812701702, -0.122091334, -2.01162417, 1.25854853, 3.0533095, 2.01197383, 1.07812134, 3.09780364, -0.349561988, 2.43573659, 0.314918803, -0.988548213, -0.188247204, 2.47305654, -2.9914337, 1.47154532, -0.66115141, -1.68885773, -0.30927999, -2.81524886, -1.7522019, -1.69230287, -0.000752950306])
    actual = corner_orientations(img, corners, octagon(3, 2))
    assert_almost_equal(actual, expected)