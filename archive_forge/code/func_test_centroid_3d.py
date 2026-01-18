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
def test_centroid_3d():
    centroid = regionprops(SAMPLE_3D)[0].centroid
    assert_array_almost_equal(centroid, (1.66666667, 1.55555556, 1.55555556))
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=(1, 1, 1))
    cZ = Mpqr(1, 0, 0) / Mpqr(0, 0, 0)
    cY = Mpqr(0, 1, 0) / Mpqr(0, 0, 0)
    cX = Mpqr(0, 0, 1) / Mpqr(0, 0, 0)
    assert_array_almost_equal((cZ, cY, cX), centroid)