import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
@run_in_parallel()
def test_line_horizontal():
    img = np.zeros((10, 10))
    rr, cc = line(0, 0, 0, 9)
    img[rr, cc] = 1
    img_ = np.zeros((10, 10))
    img_[0, :] = 1
    assert_array_equal(img, img_)