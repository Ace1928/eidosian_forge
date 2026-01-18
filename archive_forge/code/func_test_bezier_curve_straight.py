import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_bezier_curve_straight():
    image = np.zeros((200, 200), dtype=int)
    r0, c0 = (50, 50)
    r1, c1 = (150, 50)
    r2, c2 = (150, 150)
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 0)
    image[rr, cc] = 1
    image2 = np.zeros((200, 200), dtype=int)
    rr, cc = line(r0, c0, r2, c2)
    image2[rr, cc] = 1
    assert_array_equal(image, image2)