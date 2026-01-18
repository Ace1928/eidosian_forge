import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64])
def test_fill_hole(dtype):
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = np.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0], dtype=dtype)
    mask = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0], dtype=dtype)
    result = reconstruction(seed, mask, method='erosion')
    assert result.dtype == _supported_float_type(mask.dtype)
    expected = np.array([0, 3, 6, 4, 4, 4, 4, 4, 2, 0], dtype=dtype)
    assert_array_almost_equal(result, expected)