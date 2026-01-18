import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
def test_isodata_moon_image_negative_int():
    moon = util.img_as_ubyte(data.moon()).astype(np.int32)
    moon -= 100
    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == -14
    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [-14, -13, -12, 22, 23, 24, 39, 40])