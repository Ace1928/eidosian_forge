import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
@pytest.mark.parametrize('mode', ['thick', 'inner', 'outer', 'subpixel'])
def test_boundaries_constant_image(mode):
    """A constant-valued image has not boundaries."""
    ones = np.ones((8, 8), dtype=int)
    b = find_boundaries(ones, mode=mode)
    assert np.all(b == 0)