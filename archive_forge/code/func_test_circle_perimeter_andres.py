import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_circle_perimeter_andres():
    img = np.zeros((15, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 0, method='andres')
    img[rr, cc] = 1
    assert np.sum(img) == 1
    assert img[7][7] == 1
    img = np.zeros((17, 15), 'uint8')
    rr, cc = circle_perimeter(7, 7, 7, method='andres')
    img[rr, cc] = 1
    img_ = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert_array_equal(img, img_)