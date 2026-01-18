import numpy as np
from skimage import data
from skimage.segmentation import felzenszwalb
from skimage._shared import testing
from skimage._shared.testing import (
def test_merging():
    img = np.array([[0, 0.3], [0.7, 1]])
    seg = felzenszwalb(img, scale=0, sigma=0, min_size=2)
    assert_equal(len(np.unique(seg)), 2)
    assert_array_equal(seg[0, :], 0)
    assert_array_equal(seg[1, :], 1)