import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_matrix_scalar(self):
    r = np.r_['r', [1, 2], 3]
    assert_equal(type(r), np.matrix)
    assert_equal(np.array(r), [[1, 2, 3]])