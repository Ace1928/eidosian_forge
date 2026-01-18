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
def test_multiotsu_output():
    image = np.zeros((100, 100), dtype='int')
    coords = [(25, 25), (50, 50), (75, 75)]
    values = [64, 128, 192]
    for coor, val in zip(coords, values):
        rr, cc = disk(coor, 20)
        image[rr, cc] = val
    thresholds = [0, 64, 128]
    assert np.array_equal(thresholds, threshold_multiotsu(image, classes=4))