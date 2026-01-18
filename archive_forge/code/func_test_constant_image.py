import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_constant_image(self):
    image = np.full((20, 20), 128, dtype=np.uint8)
    peaks = peak.peak_local_max(image, min_distance=1)
    assert len(peaks) == 0