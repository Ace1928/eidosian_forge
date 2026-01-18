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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_log_warp_polar(dtype):
    radii = [np.exp(2), np.exp(3), np.exp(4), np.exp(5), np.exp(5) - 1, np.exp(5) + 1]
    radii = [int(x) for x in radii]
    image = np.zeros([301, 301])
    for rad in radii:
        rr, cc, val = circle_perimeter_aa(150, 150, rad)
        image[rr, cc] = val
    image = image.astype(dtype, copy=False)
    warped = warp_polar(image, radius=200, scaling='log')
    assert warped.dtype == _supported_float_type(dtype)
    profile = warped.mean(axis=0)
    peaks_coord = peak_local_max(profile)
    peaks_coord.sort(axis=0)
    gaps = peaks_coord[1:] - peaks_coord[:-1]
    assert all((x >= 38 and x <= 40 for x in gaps))