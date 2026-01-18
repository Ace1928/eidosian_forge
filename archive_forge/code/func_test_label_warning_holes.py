import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_label_warning_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]], dtype=int)
    with expected_warnings(['use a boolean array?']):
        remove_small_holes(labeled_holes_image, area_threshold=3)
    remove_small_holes(labeled_holes_image.astype(bool), area_threshold=3)