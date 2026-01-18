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
@pytest.mark.parametrize('image', [[0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0.1, 0, 0, 0.1, 0, 0.1], [0, 0, 0.1, 0, 0, 0.1, 0.01, 0.1], [0, 0, 1, 0, 0, 1, 0.5, 1], [1, 1], [1, 2], [0, 254, 255], [0, 1, 255], [0.1, 0.8, 0.9]])
def test_li_pathological(image):
    image = np.array(image)
    threshold = threshold_li(image)
    assert np.isfinite(threshold)