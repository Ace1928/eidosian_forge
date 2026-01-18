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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_adjust_gamma_zero(dtype):
    """White image should be returned for gamma equal to zero"""
    image = np.random.uniform(0, 255, (8, 8)).astype(dtype, copy=False)
    result = exposure.adjust_gamma(image, 0)
    dtype = image.dtype.type
    assert_array_equal(result, dtype_range[dtype][1])
    assert result.dtype == image.dtype