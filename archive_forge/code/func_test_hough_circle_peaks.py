import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_circle_peaks():
    x_0, y_0, rad_0 = (99, 50, 20)
    img = np.zeros((120, 100), dtype=int)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1
    x_1, y_1, rad_1 = (49, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    img[x, y] = 1
    radii = [rad_0, rad_1]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf, total_num_peaks=np.inf)
    s = np.argsort(out[3])
    assert_equal(out[1][s], np.array([y_0, y_1]))
    assert_equal(out[2][s], np.array([x_0, x_1]))
    assert_equal(out[3][s], np.array([rad_0, rad_1]))