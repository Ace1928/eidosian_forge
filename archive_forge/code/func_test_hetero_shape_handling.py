import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_hetero_shape_handling(self):
    a = np.zeros((3, 3, 7, 3), int)
    with assert_raises_regex(ValueError, 'equal length'):
        fill_diagonal(a, 2)