import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_moments_spacing():
    spacing = (2, 0.3)
    m = regionprops(SAMPLE, spacing=spacing)[0].moments
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    assert_almost_equal(m[0, 0], Mpq(0, 0))
    assert_almost_equal(m[0, 1], Mpq(0, 1))
    assert_almost_equal(m[0, 2], Mpq(0, 2))
    assert_almost_equal(m[0, 3], Mpq(0, 3))
    assert_almost_equal(m[1, 0], Mpq(1, 0))
    assert_almost_equal(m[1, 1], Mpq(1, 1))
    assert_almost_equal(m[1, 2], Mpq(1, 2))
    assert_almost_equal(m[2, 0], Mpq(2, 0))
    assert_almost_equal(m[2, 1], Mpq(2, 1))
    assert_almost_equal(m[3, 0], Mpq(3, 0))