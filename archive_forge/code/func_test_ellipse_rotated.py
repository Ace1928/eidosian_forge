import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_rotated():
    img = np.zeros((1000, 1200), dtype=np.uint8)
    for rot in range(0, 180, 10):
        img.fill(0)
        angle = np.deg2rad(rot)
        rr, cc = ellipse(500, 600, 200, 400, rotation=angle)
        img[rr, cc] = 1
        angle_estim_raw = regionprops(img)[0].orientation
        angle_estim = np.round(angle_estim_raw, 3) % (np.pi / 2)
        assert_almost_equal(angle_estim, angle % (np.pi / 2), 2)