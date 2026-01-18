import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks_zero_input():
    img = np.zeros((100, 100), dtype='uint8')
    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    h, a, d = transform.hough_line_peaks(hspace, angles, dists)
    assert_equal(a, np.array([]))