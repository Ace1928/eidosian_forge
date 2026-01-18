import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_error_shape_mismatch(self):
    x = np.zeros((3, 3, 2, 3), int)
    with assert_raises_regex(ValueError, 'equal length'):
        diag_indices_from(x)