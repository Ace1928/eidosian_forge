import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_perimeter_dot_nzeroangle():
    img = np.zeros((30, 15), 'uint8')
    rr, cc = ellipse_perimeter(15, 7, 0, 0, 1)
    img[rr, cc] = 1
    assert np.sum(img) == 1
    assert img[15][7] == 1