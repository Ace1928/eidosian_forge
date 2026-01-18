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
def test_adjust_sigmoid_cutoff_zero():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to zero and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[127, 137, 147, 156, 166, 175, 183, 191], [198, 205, 211, 216, 221, 225, 229, 232], [235, 238, 240, 242, 244, 245, 247, 248], [249, 250, 250, 251, 251, 252, 252, 253], [253, 253, 253, 253, 254, 254, 254, 254], [254, 254, 254, 254, 254, 254, 254, 254], [254, 254, 254, 254, 254, 254, 254, 254], [254, 254, 254, 254, 254, 254, 254, 254]], dtype=np.uint8)
    result = exposure.adjust_sigmoid(image, 0, 10)
    assert_array_equal(result, expected)