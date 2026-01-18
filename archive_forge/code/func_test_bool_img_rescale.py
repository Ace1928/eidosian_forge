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
def test_bool_img_rescale():
    img = np.ones((12, 18), dtype=bool)
    img[2:-2, 4:-4] = False
    res = rescale(img, 0.5)
    expected = np.ones((6, 9))
    expected[1:-1, 2:-2] = False
    assert_array_equal(res, expected)