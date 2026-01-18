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
def test_hsv2rgb_dtype(self):
    rgb = self.img_rgb.astype('float32')[::16, ::16]
    hsv = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2]) for pt in rgb.reshape(-1, 3)], dtype='float64').reshape(rgb.shape)
    hsv32 = hsv.astype('float32')
    assert hsv2rgb(hsv).dtype == hsv.dtype
    assert hsv2rgb(hsv32).dtype == hsv32.dtype