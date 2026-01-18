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
def test_multiotsu_lut():
    for classes in [2, 3, 4]:
        for name in ['camera', 'moon', 'coins', 'text', 'clock', 'page']:
            img = getattr(data, name)()
            prob, bin_centers = histogram(img.ravel(), nbins=256, source_range='image', normalize=True)
            prob = prob.astype('float32')
            result_lut = _get_multiotsu_thresh_indices_lut(prob, classes - 1)
            result = _get_multiotsu_thresh_indices(prob, classes - 1)
            assert np.array_equal(result_lut, result)