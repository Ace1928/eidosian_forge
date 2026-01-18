import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
@pytest.mark.parametrize('mode', ['clip', 'wrap', 'raise'])
def test_empty_array_ravel(self, mode):
    res = np.ravel_multi_index(np.zeros((3, 0), dtype=np.intp), (2, 1, 0), mode=mode)
    assert res.shape == (0,)
    with assert_raises(ValueError):
        np.ravel_multi_index(np.zeros((3, 1), dtype=np.intp), (2, 1, 0), mode=mode)