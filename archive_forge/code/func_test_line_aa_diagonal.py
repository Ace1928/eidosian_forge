import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_line_aa_diagonal():
    img = np.zeros((10, 10))
    rr, cc, val = line_aa(0, 0, 9, 6)
    img[rr, cc] = 1
    r, c = line(0, 0, 9, 6)
    for r_i, c_i in zip(r, c):
        assert_equal(img[r_i, c_i], 1)