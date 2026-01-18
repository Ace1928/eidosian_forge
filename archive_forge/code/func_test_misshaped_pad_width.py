import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('pad_width', [(4, 5, 6, 7), ((1,), (2,), (3,)), ((1, 2), (3, 4), (5, 6)), ((3, 4, 5), (0, 1, 2))])
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_misshaped_pad_width(self, pad_width, mode):
    arr = np.arange(30).reshape((6, 5))
    match = 'operands could not be broadcast together'
    with pytest.raises(ValueError, match=match):
        np.pad(arr, pad_width, mode)