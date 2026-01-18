import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_circle_peaks_total_peak_and_min_distance():
    img = np.zeros((120, 120), dtype=int)
    cx = cy = [40, 50, 60, 70, 80]
    radii = range(20, 30, 2)
    for i in range(len(cx)):
        y, x = circle_perimeter(cy[i], cx[i], radii[i])
        img[x, y] = 1
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=15, min_ydistance=15, threshold=None, num_peaks=np.inf, total_num_peaks=2, normalize=True)
    assert_equal(out[1], np.array(cy[:4:2]))
    assert_equal(out[2], np.array(cx[:4:2]))
    assert_equal(out[3], np.array(radii[:4:2]))