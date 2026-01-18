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
def test_is_low_contrast():
    image = np.linspace(0, 0.04, 100)
    assert exposure.is_low_contrast(image)
    image[-1] = 1
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)
    image = (image * 255).astype(np.uint8)
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)
    image = image.astype(np.uint16) * 2 ** 8
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)