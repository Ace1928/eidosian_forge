import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('func', [corner_moravec, corner_harris, corner_shi_tomasi, corner_kitchen_rosenfeld])
def test_corner_dtype(dtype, func):
    im = np.zeros((50, 50), dtype=dtype)
    im[:25, :25] = 1.0
    out_dtype = _supported_float_type(dtype)
    corners = func(im)
    assert corners.dtype == out_dtype