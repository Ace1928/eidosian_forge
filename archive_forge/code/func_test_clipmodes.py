import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_clipmodes(self):
    assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode='wrap'), np.ravel_multi_index([1, 1, 6, 2], (4, 3, 7, 12)))
    assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode=('wrap', 'raise', 'clip', 'raise')), np.ravel_multi_index([1, 1, 0, 2], (4, 3, 7, 12)))
    assert_raises(ValueError, np.ravel_multi_index, [5, 1, -1, 2], (4, 3, 7, 12))