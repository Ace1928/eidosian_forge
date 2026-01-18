import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_labeled_image():
    labeled_image = np.array([[2, 2, 2, 0, 1], [2, 2, 2, 0, 1], [2, 0, 0, 0, 0], [0, 0, 3, 3, 3]], dtype=int)
    expected = np.array([[2, 2, 2, 0, 0], [2, 2, 2, 0, 0], [2, 0, 0, 0, 0], [0, 0, 3, 3, 3]], dtype=int)
    observed = remove_small_objects(labeled_image, min_size=3)
    assert_array_equal(observed, expected)