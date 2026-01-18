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
def test_multiotsu_hist_parameter():
    for classes in [2, 3, 4]:
        for name in ['camera', 'moon', 'coins', 'text', 'clock', 'page']:
            img = getattr(data, name)()
            sk_hist = histogram(img, nbins=256)
            thresh_img = threshold_multiotsu(img, classes)
            thresh_sk_hist = threshold_multiotsu(classes=classes, hist=sk_hist)
            assert np.allclose(thresh_img, thresh_sk_hist)