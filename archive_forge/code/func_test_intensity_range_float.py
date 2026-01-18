import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('test_input,expected', [('image', [0.1, 0.2]), ('dtype', [-1, 1]), ((0.3, 0.4), [0.3, 0.4])])
def test_intensity_range_float(test_input, expected):
    image = np.array([0.1, 0.2], dtype=np.float64)
    out = intensity_range(image, range_values=test_input)
    assert_array_equal(out, expected)