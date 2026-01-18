import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_morphsnakes_incorrect_ndim():
    img = np.zeros((4, 4, 4, 4))
    ls = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError):
        morphological_chan_vese(img, num_iter=1, init_level_set=ls)
    with pytest.raises(ValueError):
        morphological_geodesic_active_contour(img, num_iter=1, init_level_set=ls)