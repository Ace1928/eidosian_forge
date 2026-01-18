import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
@run_in_parallel()
def test_hough_line():
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1
    out, angles, d = transform.hough_line(img)
    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]
    assert_almost_equal(dist, 80.0, 1)
    assert_almost_equal(theta, 1.41, 1)