import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks_dist():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = transform.hough_line(img)
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_distance=5)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_distance=15)[0]) == 1