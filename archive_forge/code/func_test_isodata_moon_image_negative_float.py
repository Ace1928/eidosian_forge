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
def test_isodata_moon_image_negative_float():
    moon = util.img_as_ubyte(data.moon()).astype(np.float64)
    moon -= 100
    assert -14 < threshold_isodata(moon) < -13
    thresholds = threshold_isodata(moon, return_all=True)
    assert_almost_equal(thresholds, [-13.83789062, -12.84179688, -11.84570312, 22.02148438, 23.01757812, 24.01367188, 38.95507812, 39.95117188])