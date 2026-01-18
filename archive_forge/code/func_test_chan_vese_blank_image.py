import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_blank_image():
    img = np.zeros((10, 10))
    level_set = np.random.rand(10, 10)
    ref = level_set > 0
    result = chan_vese(img, mu=0.0, tol=0.0, init_level_set=level_set)
    assert_array_equal(result, ref)