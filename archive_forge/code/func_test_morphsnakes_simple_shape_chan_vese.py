import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_morphsnakes_simple_shape_chan_vese():
    img = gaussian_blob()
    ls1 = disk_level_set(img.shape, center=(5, 5), radius=3)
    ls2 = disk_level_set(img.shape, center=(5, 5), radius=6)
    acwe_ls1 = morphological_chan_vese(img, num_iter=10, init_level_set=ls1)
    acwe_ls2 = morphological_chan_vese(img, num_iter=10, init_level_set=ls2)
    assert_array_equal(acwe_ls1, acwe_ls2)
    assert acwe_ls1.dtype == acwe_ls2.dtype == np.int8