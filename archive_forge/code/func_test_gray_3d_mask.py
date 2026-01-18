from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_gray_3d_mask():
    msk = np.zeros((20, 21, 22))
    msk[2:-2, 2:-2, 2:-2] = 1
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 22))
    slices = []
    for dim_size in img.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    shades = np.linspace(0, 1, 8)
    for s, sh in zip(slices, shades):
        img[s] = sh
    img += 0.001 * rng.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, sigma=0, n_segments=8, channel_axis=None, convert2lab=False, mask=msk)
    assert_equal(len(np.unique(seg)), 9)
    for s, c in zip(slices, range(1, 9)):
        assert_equal(seg[s][2:-2, 2:-2, 2:-2], c)