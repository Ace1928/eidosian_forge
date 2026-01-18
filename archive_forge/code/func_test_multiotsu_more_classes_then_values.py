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
def test_multiotsu_more_classes_then_values():
    img = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=2)
    img[:, 3:] = 2
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=3)
    img[:, 6:] = 3
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=4)