import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_3d_vs_fiji():
    img = binary_blobs(32, 0.05, n_dim=3, rng=1234)
    img = img[:-2, ...]
    img_s = skeletonize(img)
    img_f = io.imread(fetch('data/_blobs_3d_fiji_skeleton.tif')).astype(bool)
    assert_equal(img_s, img_f)