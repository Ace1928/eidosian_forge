import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_rotation_symmetry():
    img1 = np.zeros((150, 150), dtype=np.uint8)
    img2 = np.zeros((150, 150), dtype=np.uint8)
    for angle in range(0, 180, 15):
        img1.fill(0)
        rr, cc = ellipse(80, 70, 60, 40, rotation=np.deg2rad(angle))
        img1[rr, cc] = 1
        img2.fill(0)
        rr, cc = ellipse(80, 70, 60, 40, rotation=np.deg2rad(angle + 180))
        img2[rr, cc] = 1
        assert_array_equal(img1, img2)