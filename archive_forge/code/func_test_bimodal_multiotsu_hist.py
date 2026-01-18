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
def test_bimodal_multiotsu_hist():
    for name in ['camera', 'moon', 'coins', 'text', 'clock', 'page']:
        img = getattr(data, name)()
        assert threshold_otsu(img) == threshold_multiotsu(img, 2)
    for name in ['chelsea', 'coffee', 'astronaut', 'rocket']:
        img = rgb2gray(getattr(data, name)())
        assert threshold_otsu(img) == threshold_multiotsu(img, 2)