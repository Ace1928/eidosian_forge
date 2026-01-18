import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def gaussian_blob():
    coords = np.mgrid[-5:6, -5:6]
    sqrdistances = (coords ** 2).sum(0)
    return np.exp(-sqrdistances / 10)