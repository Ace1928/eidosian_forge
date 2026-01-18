import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.ndimage import map_coordinates
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color.colorconv import rgb2gray
from skimage.data import checkerboard, astronaut
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage.transform._warps import (
from skimage.transform._geometric import (
from skimage.util.dtype import img_as_float, _convert
def test_log_polar_mapping():
    output_coords = np.array([[0, 0], [0, 90], [0, 180], [0, 270], [99, 0], [99, 180], [99, 270], [99, 45]])
    ground_truth = np.array([[101, 100], [100, 101], [99, 100], [100, 99], [195.4992586, 100], [4.5007414, 100], [100, 4.5007414], [167.52817336, 167.52817336]])
    k_angle = 360 / (2 * np.pi)
    k_radius = 100 / np.log(100)
    center = (100, 100)
    coords = _log_polar_mapping(output_coords, k_angle, k_radius, center)
    assert np.allclose(coords, ground_truth)