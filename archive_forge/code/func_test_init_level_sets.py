import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_init_level_sets():
    image = np.zeros((6, 6))
    checkerboard_ls = morphological_chan_vese(image, 0, 'checkerboard')
    checkerboard_ref = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0]], dtype=np.int8)
    disk_ls = morphological_geodesic_active_contour(image, 0, 'disk')
    disk_ref = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0]], dtype=np.int8)
    assert_array_equal(checkerboard_ls, checkerboard_ref)
    assert_array_equal(disk_ls, disk_ref)