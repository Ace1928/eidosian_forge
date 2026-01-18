import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_disk(self):
    """regression test of img-1194, footprint = [1]
        Test peak.peak_local_max when every point is a local maximum
        """
    image = np.random.uniform(size=(10, 20))
    footprint = np.array([[1]])
    peak_idx = peak.peak_local_max(image, labels=np.ones((10, 20), int), footprint=footprint, min_distance=1, threshold_rel=0, threshold_abs=-1, exclude_border=False)
    result = np.zeros_like(image, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert np.all(result)
    peak_idx = peak.peak_local_max(image, footprint=footprint, threshold_abs=-1, exclude_border=False)
    result = np.zeros_like(image, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert np.all(result)