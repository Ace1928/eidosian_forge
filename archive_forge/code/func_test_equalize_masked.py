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
def test_equalize_masked():
    img = util.img_as_float(test_img)
    mask = np.zeros(test_img.shape)
    mask[100:400, 100:400] = 1
    img_mask_eq = exposure.equalize_hist(img, mask=mask)
    img_eq = exposure.equalize_hist(img)
    cdf, bin_edges = exposure.cumulative_distribution(img_mask_eq)
    check_cdf_slope(cdf)
    assert not (img_eq == img_mask_eq).all()