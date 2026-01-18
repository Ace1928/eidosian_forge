import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_circle_extended():
    img = np.zeros((100, 100), dtype=int)
    radius = 20
    x_0, y_0 = (-5, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x[np.where(x > 0)], y[np.where(x > 0)]] = 1
    out = transform.hough_circle(img, np.array([radius], dtype=np.intp), full_output=True)
    x, y = np.where(out[0] == out[0].max())
    assert_equal(x[0], x_0 + radius)
    assert_equal(y[0], y_0 + radius)