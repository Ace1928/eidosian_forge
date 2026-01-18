import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks_ordered():
    testim = np.zeros((256, 64), dtype=bool)
    testim[50:100, 20] = True
    testim[20:225, 25] = True
    testim[15:35, 50] = True
    testim[1:-1, 58] = True
    hough_space, angles, dists = transform.hough_line(testim)
    hspace, _, _ = transform.hough_line_peaks(hough_space, angles, dists)
    assert hspace[0] > hspace[1]