from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_color_3d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 22, 3))
    slices = []
    for dim_size in img.shape[:-1]:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    colors = list(product(*([0, 1],) * 3))
    for s, c in zip(slices, colors):
        img[s] = c
    img += 0.01 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=8, start_label=0)
    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_equal(seg[s], c)