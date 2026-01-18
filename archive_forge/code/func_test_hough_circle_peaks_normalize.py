import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_circle_peaks_normalize():
    x_0, y_0, rad_0 = (50, 50, 20)
    img = np.zeros((120, 100), dtype=int)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1
    x_1, y_1, rad_1 = (60, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    img[x, y] = 1
    radii = [rad_0, rad_1]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=15, min_ydistance=15, threshold=None, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False)
    assert_equal(out[1], np.array([y_1]))
    assert_equal(out[2], np.array([x_1]))
    assert_equal(out[3], np.array([rad_1]))