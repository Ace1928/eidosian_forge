import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_morphsnakes_simple_shape_geodesic_active_contour():
    img = disk_level_set((11, 11), center=(5, 5), radius=3.5).astype(float)
    gimg = inverse_gaussian_gradient(img, alpha=10.0, sigma=1.0)
    ls = disk_level_set(img.shape, center=(5, 5), radius=6)
    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    gac_ls = morphological_geodesic_active_contour(gimg, num_iter=10, init_level_set=ls, balloon=-1)
    assert_array_equal(gac_ls, ref)
    assert gac_ls.dtype == np.int8