import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', ['mean', 'median', 'minimum', 'maximum'])
@pytest.mark.parametrize('stat_length', [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))])
def test_check_negative_stat_length(self, mode, stat_length):
    arr = np.arange(30).reshape((6, 5))
    match = "index can't contain negative values"
    with pytest.raises(ValueError, match=match):
        np.pad(arr, 2, mode, stat_length=stat_length)