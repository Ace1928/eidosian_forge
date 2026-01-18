import numpy as np
import pytest
from skimage.segmentation import quickshift
from skimage._shared import testing
from skimage._shared.testing import (
@run_in_parallel()
@testing.parametrize('dtype', [np.float32, np.float64])
def test_grey(dtype):
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21))
    img[:10, 10:] = 0.2
    img[10:, :10] = 0.4
    img[10:, 10:] = 0.6
    img += 0.05 * rng.normal(size=img.shape)
    img = img.astype(dtype, copy=False)
    seg = quickshift(img, kernel_size=2, max_dist=3, rng=0, convert2lab=False, sigma=0)
    quickshift(img, kernel_size=2, max_dist=3, rng=0, convert2lab=False, sigma=0)
    assert_equal(len(np.unique(seg)), 4)
    for i in range(4):
        hist = np.histogram(img[seg == i], bins=[0, 0.1, 0.3, 0.5, 1])[0]
        assert_greater(hist[i], 20)