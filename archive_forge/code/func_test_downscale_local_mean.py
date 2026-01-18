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
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_downscale_local_mean(dtype):
    image1 = np.arange(4 * 6, dtype=dtype).reshape(4, 6)
    out1 = downscale_local_mean(image1, (2, 3))
    float_dtype = dtype if np.dtype(dtype).kind == 'f' else np.float64
    assert out1.dtype == float_dtype
    expected1 = np.array([[4.0, 7.0], [16.0, 19.0]])
    assert_array_equal(expected1, out1)
    image2 = np.arange(5 * 8, dtype=dtype).reshape(5, 8)
    out2 = downscale_local_mean(image2, (4, 5))
    assert out2.dtype == float_dtype
    expected2 = np.array([[14.0, 10.8], [8.5, 5.7]])
    rtol = 0.001 if dtype == np.float16 else 1e-07
    assert_allclose(expected2, out2, rtol=rtol)