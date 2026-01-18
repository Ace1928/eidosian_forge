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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_lab_lch_roundtrip_dtypes(dtype):
    rgb = img_as_float(data.colorwheel()).astype(dtype=dtype, copy=False)
    lab = rgb2lab(rgb)
    float_dtype = _supported_float_type(dtype)
    assert lab.dtype == float_dtype
    lab2 = lch2lab(lab2lch(lab))
    decimal = 4 if float_dtype == np.float32 else 7
    assert_array_almost_equal(lab2, lab, decimal=decimal)