import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_circle_perimeter_aa():
    img = np.zeros((15, 15), 'uint8')
    rr, cc, val = circle_perimeter_aa(7, 7, 0)
    img[rr, cc] = 1
    assert np.sum(img) == 1
    assert img[7][7] == 1
    img = np.zeros((17, 17), 'uint8')
    rr, cc, val = circle_perimeter_aa(8, 8, 7)
    img[rr, cc] = val * 255
    img_ = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 82, 180, 236, 255, 236, 180, 82, 0, 0, 0, 0, 0], [0, 0, 0, 0, 189, 172, 74, 18, 0, 18, 74, 172, 189, 0, 0, 0, 0], [0, 0, 0, 229, 25, 0, 0, 0, 0, 0, 0, 0, 25, 229, 0, 0, 0], [0, 0, 189, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 189, 0, 0], [0, 82, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 82, 0], [0, 180, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 180, 0], [0, 236, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 236, 0], [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0], [0, 236, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 236, 0], [0, 180, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 180, 0], [0, 82, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 82, 0], [0, 0, 189, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 189, 0, 0], [0, 0, 0, 229, 25, 0, 0, 0, 0, 0, 0, 0, 25, 229, 0, 0, 0], [0, 0, 0, 0, 189, 172, 74, 18, 0, 18, 74, 172, 189, 0, 0, 0, 0], [0, 0, 0, 0, 0, 82, 180, 236, 255, 236, 180, 82, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert_array_equal(img, img_)