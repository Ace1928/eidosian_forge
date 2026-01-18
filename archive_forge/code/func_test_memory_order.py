import numpy as np
from skimage.measure import find_contours
from skimage._shared.testing import assert_array_equal
import pytest
@pytest.mark.parametrize('level', [0.5, None])
def test_memory_order(level):
    contours = find_contours(np.ascontiguousarray(r), level)
    assert len(contours) == 1
    contours = find_contours(np.asfortranarray(r), level)
    assert len(contours) == 1