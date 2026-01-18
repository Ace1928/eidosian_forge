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
def test_li_arbitrary_start_point():
    cell = data.cell()
    max_stationary_point = threshold_li(cell)
    low_stationary_point = threshold_li(cell, initial_guess=np.percentile(cell, 5))
    optimum = threshold_li(cell, initial_guess=np.percentile(cell, 95))
    assert 67 < max_stationary_point < 68
    assert 48 < low_stationary_point < 49
    assert 111 < optimum < 112