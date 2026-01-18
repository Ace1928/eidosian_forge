import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_input_values_with_labels():
    img = np.random.rand(128, 128)
    labels = np.zeros((128, 128), int)
    labels[10:20, 10:20] = 1
    labels[12:16, 12:16] = 0
    img_before = img.copy()
    _ = peak.peak_local_max(img, labels=labels)
    assert_array_equal(img, img_before)