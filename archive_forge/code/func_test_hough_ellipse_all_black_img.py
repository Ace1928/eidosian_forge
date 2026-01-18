import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_ellipse_all_black_img():
    assert transform.hough_ellipse(np.zeros((100, 100))).shape == (0, 6)