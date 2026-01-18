import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_laplace_zeros():
    """Laplace on a square image."""
    image = np.zeros((9, 9))
    image[3:-3, 3:-3] = 1
    result = filters.laplace(image)
    check_result = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0, 1.0, 2.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0, 1.0, 2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert_allclose(result, check_result)