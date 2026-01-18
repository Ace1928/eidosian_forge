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
def test_equalize_ubyte():
    img = util.img_as_ubyte(test_img)
    img_eq = exposure.equalize_hist(img)
    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)