import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_ellipse_non_zero_posangle3():
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35 + np.pi / 2.0
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]), orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)