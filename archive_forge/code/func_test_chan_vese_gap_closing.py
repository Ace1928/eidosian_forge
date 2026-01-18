import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_gap_closing():
    ref = np.zeros((20, 20))
    ref[8:15, :] = np.ones((7, 20))
    img = ref.copy()
    img[:, 6] = np.zeros(20)
    result = chan_vese(img, mu=0.7, tol=0.001, max_num_iter=1000, dt=1000, init_level_set='disk').astype(float)
    assert_array_equal(result, ref)