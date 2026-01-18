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
def test_all_negative_image():
    im = np.array([-100, -1], dtype=np.int8)
    frequencies, bin_centers = exposure.histogram(im)
    assert_array_equal(bin_centers, np.arange(-100, 0))
    assert frequencies[0] == 1
    assert frequencies[-1] == 1
    assert_array_equal(frequencies[1:-1], 0)