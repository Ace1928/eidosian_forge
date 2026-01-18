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
def test_adapthist_constant():
    """Test constant image, float and uint"""
    img = np.zeros((8, 8))
    img += 2
    img = img.astype(np.uint16)
    adapted = exposure.equalize_adapthist(img, 3)
    assert np.min(adapted) == np.max(adapted)
    img = np.zeros((8, 8))
    img += 0.1
    img = img.astype(np.float64)
    adapted = exposure.equalize_adapthist(img, 3)
    assert np.min(adapted) == np.max(adapted)