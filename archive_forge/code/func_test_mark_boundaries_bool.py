import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
def test_mark_boundaries_bool():
    image = np.zeros((10, 10), dtype=bool)
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1
    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)