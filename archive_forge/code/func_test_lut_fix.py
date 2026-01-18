import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
def test_lut_fix(self):
    im = np.zeros((6, 6), dtype=bool)
    im[1, 2] = 1
    im[2, 2] = 1
    im[2, 3] = 1
    im[3, 3] = 1
    im[3, 4] = 1
    im[4, 4] = 1
    im[4, 5] = 1
    result = skeletonize(im)
    expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=bool)
    assert np.all(result == expected)