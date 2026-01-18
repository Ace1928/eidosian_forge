import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_morphsnakes_black():
    img = np.zeros((11, 11))
    ls = disk_level_set(img.shape, center=(5, 5), radius=3)
    ref_zeros = np.zeros(img.shape, dtype=np.int8)
    ref_ones = np.ones(img.shape, dtype=np.int8)
    acwe_ls = morphological_chan_vese(img, num_iter=6, init_level_set=ls)
    assert_array_equal(acwe_ls, ref_zeros)
    gac_ls = morphological_geodesic_active_contour(img, num_iter=6, init_level_set=ls)
    assert_array_equal(gac_ls, ref_zeros)
    gac_ls2 = morphological_geodesic_active_contour(img, num_iter=6, init_level_set=ls, balloon=1, threshold=-1, smoothing=0)
    assert_array_equal(gac_ls2, ref_ones)
    assert acwe_ls.dtype == gac_ls.dtype == gac_ls2.dtype == np.int8