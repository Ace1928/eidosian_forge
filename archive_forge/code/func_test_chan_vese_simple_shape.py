import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
def test_chan_vese_simple_shape():
    img = np.zeros((10, 10))
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=1e-08).astype(float)
    assert_array_equal(result, img)