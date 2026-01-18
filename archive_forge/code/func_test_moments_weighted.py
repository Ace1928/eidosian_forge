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
def test_moments_weighted():
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted
    ref = np.array([[74.0, 699.0, 7863.0, 97317.0], [410.0, 3785.0, 44063.0, 572567.0], [2750.0, 24855.0, 293477.0, 3900717.0], [19778.0, 175001.0, 2081051.0, 28078871.0]])
    assert_array_almost_equal(wm, ref)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(Mpq(0, 0), ref[0, 0])
    assert_almost_equal(Mpq(0, 1), ref[0, 1])
    assert_almost_equal(Mpq(0, 2), ref[0, 2])
    assert_almost_equal(Mpq(0, 3), ref[0, 3])
    assert_almost_equal(Mpq(1, 0), ref[1, 0])
    assert_almost_equal(Mpq(1, 1), ref[1, 1])
    assert_almost_equal(Mpq(1, 2), ref[1, 2])
    assert_almost_equal(Mpq(1, 3), ref[1, 3])
    assert_almost_equal(Mpq(2, 0), ref[2, 0])
    assert_almost_equal(Mpq(2, 1), ref[2, 1])
    assert_almost_equal(Mpq(2, 2), ref[2, 2])
    assert_almost_equal(Mpq(2, 3), ref[2, 3])
    assert_almost_equal(Mpq(3, 0), ref[3, 0])
    assert_almost_equal(Mpq(3, 1), ref[3, 1])
    assert_almost_equal(Mpq(3, 2), ref[3, 2])
    assert_almost_equal(Mpq(3, 3), ref[3, 3])