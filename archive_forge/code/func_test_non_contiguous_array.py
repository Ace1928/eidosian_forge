import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_non_contiguous_array(mode):
    arr = np.arange(24).reshape(4, 6)[::2, ::2]
    result = np.pad(arr, (2, 3), mode)
    assert result.shape == (7, 8)
    assert_equal(result[2:-3, 2:-3], arr)