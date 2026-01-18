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
def test_multichannel_int_range_image():
    im = np.array([[10, 5], [100, 102]], dtype=np.int8)
    frequencies, bin_centers = exposure.histogram(im, channel_axis=-1)
    for ch in range(im.shape[-1]):
        assert_equal(len(frequencies[ch]), len(bin_centers))
    assert_equal(bin_centers[0], 5)
    assert_equal(bin_centers[-1], 102)