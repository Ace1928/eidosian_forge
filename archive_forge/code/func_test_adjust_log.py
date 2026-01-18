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
def test_adjust_log():
    """Verifying the output with expected results for logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 5, 11, 16, 22, 27, 33, 38], [43, 48, 53, 58, 63, 68, 73, 77], [82, 86, 91, 95, 100, 104, 109, 113], [117, 121, 125, 129, 133, 137, 141, 145], [149, 153, 157, 160, 164, 168, 172, 175], [179, 182, 186, 189, 193, 196, 199, 203], [206, 209, 213, 216, 219, 222, 225, 228], [231, 234, 238, 241, 244, 246, 249, 252]], dtype=np.uint8)
    result = exposure.adjust_log(image, 1)
    assert_array_equal(result, expected)