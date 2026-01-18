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
def test_intensity_range_clipped_float():
    image = np.array([0.1, 0.2], dtype=np.float64)
    out = intensity_range(image, range_values='dtype', clip_negative=True)
    assert_array_equal(out, (0, 1))