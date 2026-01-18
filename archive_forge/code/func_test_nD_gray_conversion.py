import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
@pytest.mark.parametrize('func', [rgb2gray, gray2rgb, gray2rgba])
@pytest.mark.parametrize('shape', [(3,), (2, 3), (4, 5, 3), (5, 4, 5, 3), (4, 5, 4, 5, 3)])
def test_nD_gray_conversion(func, shape):
    img = np.random.rand(*shape)
    out = func(img)
    common_ndim = min(out.ndim, len(shape))
    assert out.shape[:common_ndim] == shape[:common_ndim]