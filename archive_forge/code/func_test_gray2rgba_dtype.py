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
def test_gray2rgba_dtype():
    img_f64 = np.random.random((5, 5))
    img_f32 = img_f64.astype('float32')
    img_u8 = img_as_ubyte(img_f64)
    img_int = img_u8.astype(int)
    for img in [img_f64, img_f32, img_u8, img_int]:
        assert gray2rgba(img).dtype == img.dtype