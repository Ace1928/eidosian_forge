import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_isolated_peaks(self):
    image = np.zeros((15, 15))
    x0, y0, i0 = (12, 8, 1)
    x1, y1, i1 = (2, 2, 1)
    x2, y2, i2 = (5, 13, 1)
    image[y0, x0] = i0
    image[y1, x1] = i1
    image[y2, x2] = i2
    out = peak._prominent_peaks(image)
    assert len(out[0]) == 3
    for i, x, y in zip(out[0], out[1], out[2]):
        assert i in (i0, i1, i2)
        assert x in (x0, x1, x2)
        assert y in (y0, y1, y2)