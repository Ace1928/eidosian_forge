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
def test_moments_central_spacing():
    spacing = (1.8, 0.8)
    centralMpq = get_central_moment_function(SAMPLE, spacing=spacing)
    mu = regionprops(SAMPLE, spacing=spacing)[0].moments_central
    assert_almost_equal(mu[2, 0], centralMpq(2, 0))
    assert_almost_equal(mu[3, 0], centralMpq(3, 0))
    assert_almost_equal(mu[1, 1], centralMpq(1, 1))
    assert_almost_equal(mu[2, 1], centralMpq(2, 1))
    assert_almost_equal(mu[0, 2], centralMpq(0, 2))
    assert_almost_equal(mu[1, 2], centralMpq(1, 2))
    assert_almost_equal(mu[0, 3], centralMpq(0, 3))