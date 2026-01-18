import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
def test_find_boundaries():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:7, 2:7] = 1
    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = find_boundaries(image)
    assert_array_equal(result, ref)