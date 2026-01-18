import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_line_diag():
    img = np.zeros((5, 5))
    rr, cc = line(0, 0, 4, 4)
    img[rr, cc] = 1
    img_ = np.eye(5)
    assert_array_equal(img, img_)