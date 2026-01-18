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
@pytest.mark.parametrize('func', [rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb, rgb2rgbcie, rgbcie2rgb, xyz2lab, lab2xyz, lab2rgb, rgb2lab, xyz2luv, luv2xyz, luv2rgb, rgb2luv, lab2lch, lch2lab, rgb2yuv, yuv2rgb, rgb2yiq, yiq2rgb, rgb2ypbpr, ypbpr2rgb, rgb2ycbcr, ycbcr2rgb, rgb2ydbdr, ydbdr2rgb])
@pytest.mark.parametrize('shape', [(3,), (2, 3), (4, 5, 3), (5, 4, 5, 3), (4, 5, 4, 5, 3)])
def test_nD_color_conversion(func, shape):
    img = np.random.rand(*shape)
    out = func(img)
    assert out.shape == img.shape