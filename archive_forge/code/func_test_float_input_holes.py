import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_float_input_holes():
    float_test = np.random.rand(5, 5)
    with testing.raises(TypeError):
        remove_small_holes(float_test)