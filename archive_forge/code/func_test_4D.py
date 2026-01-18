import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_4D(self):
    image = np.zeros((30, 30, 30, 30))
    image[15, 15, 15, 15] = 1
    image[5, 5, 5, 5] = 1
    assert_array_equal(peak.peak_local_max(image, min_distance=10, threshold_rel=0), [[15, 15, 15, 15]])
    assert_array_equal(peak.peak_local_max(image, min_distance=6, threshold_rel=0), [[15, 15, 15, 15]])
    assert sorted(peak.peak_local_max(image, min_distance=10, threshold_rel=0, exclude_border=False).tolist()) == [[5, 5, 5, 5], [15, 15, 15, 15]]
    assert sorted(peak.peak_local_max(image, min_distance=5, threshold_rel=0).tolist()) == [[5, 5, 5, 5], [15, 15, 15, 15]]