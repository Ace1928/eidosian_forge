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
def test_niblack_sauvola_pathological_image():
    value = 0.03082192 + 2.19178082e-09
    src_img = np.full((4, 4), value).astype(np.float64)
    assert not np.any(np.isnan(threshold_niblack(src_img)))