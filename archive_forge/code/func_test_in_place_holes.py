import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_in_place_holes():
    image = test_holes_image.copy()
    observed = remove_small_holes(image, area_threshold=3, out=image)
    assert_equal(observed is image, True, 'remove_small_holes in_place argument failed.')