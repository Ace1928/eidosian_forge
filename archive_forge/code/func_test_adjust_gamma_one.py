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
def test_adjust_gamma_one():
    """Same image should be returned for gamma equal to one"""
    image = np.arange(0, 256, dtype=np.uint8).reshape((16, 16))
    result = exposure.adjust_gamma(image, 1)
    assert_array_equal(result, image)