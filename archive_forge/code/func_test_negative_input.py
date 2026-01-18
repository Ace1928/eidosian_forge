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
@pytest.mark.parametrize('exposure_func', [exposure.adjust_gamma, exposure.adjust_log, exposure.adjust_sigmoid])
def test_negative_input(exposure_func):
    image = np.arange(-10, 245, 4).reshape((8, 8)).astype(np.float64)
    with pytest.raises(ValueError):
        exposure_func(image)