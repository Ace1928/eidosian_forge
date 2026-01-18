import numpy as np
from skimage.measure import find_contours
from skimage._shared.testing import assert_array_equal
import pytest
def test_level_default():
    image = np.random.random((100, 100)) * 0.01 + 0.9
    contours = find_contours(image)
    assert len(contours) > 1