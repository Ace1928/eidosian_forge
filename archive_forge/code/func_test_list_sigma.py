from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_list_sigma():
    rng = np.random.default_rng(0)
    img = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], float)
    img += 0.1 * rng.normal(size=img.shape)
    result_sigma = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], int)
    with expected_warnings(['Input image is 2D: sigma number of elements must be 2']):
        seg_sigma = slic(img, n_segments=2, sigma=[1, 50, 1], channel_axis=None, start_label=0)
    assert_equal(seg_sigma, result_sigma)