from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_color_2d_mask():
    rng = np.random.default_rng(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rng.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, sigma=0, enforce_connectivity=False, mask=msk)
    assert_equal(len(np.unique(seg)), 5)
    assert_equal(seg.shape, img.shape[:-1])
    assert_equal(seg[2:10, 2:10], 1)
    assert_equal(seg[10:-2, 2:10], 4)
    assert_equal(seg[2:10, 10:-2], 2)
    assert_equal(seg[10:-2, 10:-2], 3)
    assert_equal(seg[:2, :], 0)
    assert_equal(seg[-2:, :], 0)
    assert_equal(seg[:, :2], 0)
    assert_equal(seg[:, -2:], 0)