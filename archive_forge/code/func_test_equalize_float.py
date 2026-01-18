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
def test_equalize_float(dtype):
    img = util.img_as_float(test_img).astype(dtype, copy=False)
    img_eq = exposure.equalize_hist(img)
    assert img_eq.dtype == _supported_float_type(dtype)
    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)
    assert bin_edges.dtype == _supported_float_type(dtype)