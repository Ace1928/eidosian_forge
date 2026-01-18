import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_farid_h_mask():
    """Horizontal Farid on a masked array should be zero."""
    result = filters.farid_h(np.random.uniform(size=(10, 10)), mask=np.zeros((10, 10), dtype=bool))
    assert np.all(result == 0)