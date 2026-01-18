import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_farid_v_vertical():
    """Vertical Farid on an edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.farid_v(image)
    assert np.all(result[j == 0] == result[j == 0][0])
    assert_allclose(result[np.abs(j) > 2], 0, atol=1e-10)