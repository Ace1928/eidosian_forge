from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_enforce_connectivity_mask():
    msk = np.zeros((3, 6))
    msk[:, 1:-1] = 1
    img = np.array([[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]], float)
    segments_connected = slic(img, 2, compactness=0.0001, enforce_connectivity=True, convert2lab=False, mask=msk, channel_axis=None)
    segments_disconnected = slic(img, 2, compactness=0.0001, enforce_connectivity=False, convert2lab=False, mask=msk, channel_axis=None)
    segments_connected_low_max = slic(img, 2, compactness=0.0001, enforce_connectivity=True, convert2lab=False, max_size_factor=0.8, mask=msk, channel_axis=None)
    result_connected = np.array([[0, 1, 1, 2, 2, 0], [0, 1, 1, 2, 2, 0], [0, 1, 1, 2, 2, 0]], float)
    result_disconnected = np.array([[0, 1, 1, 2, 2, 0], [0, 1, 1, 2, 2, 0], [0, 1, 1, 2, 2, 0]], float)
    assert_equal(segments_connected, result_connected)
    assert_equal(segments_disconnected, result_disconnected)
    assert_equal(segments_connected_low_max, result_connected)