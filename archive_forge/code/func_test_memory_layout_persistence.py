import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_memory_layout_persistence(mode):
    """Test if C and F order is preserved for all pad modes."""
    x = np.ones((5, 10), order='C')
    assert np.pad(x, 5, mode).flags['C_CONTIGUOUS']
    x = np.ones((5, 10), order='F')
    assert np.pad(x, 5, mode).flags['F_CONTIGUOUS']