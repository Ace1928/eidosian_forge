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
def test_gray2rgba_alpha():
    img = np.random.random((5, 5))
    img_u8 = img_as_ubyte(img)
    alpha = None
    rgba = gray2rgba(img, alpha)
    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], 1.0)
    alpha = 0.5
    rgba = gray2rgba(img, alpha)
    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], alpha)
    alpha = np.random.random((5, 5))
    rgba = gray2rgba(img, alpha)
    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], alpha)
    alpha = 0.5
    with expected_warnings(['alpha cannot be safely cast to image dtype']):
        rgba = gray2rgba(img_u8, alpha)
        assert_equal(rgba[..., :3], gray2rgb(img_u8))
    alpha = np.random.random((5, 5, 1))
    expected_err_msg = 'alpha.shape must match image.shape'
    with pytest.raises(ValueError) as err:
        rgba = gray2rgba(img, alpha)
    assert expected_err_msg == str(err.value)