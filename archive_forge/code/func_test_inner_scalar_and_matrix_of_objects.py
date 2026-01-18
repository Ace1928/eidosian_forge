import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_inner_scalar_and_matrix_of_objects():
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.inner(arr, 3), desired)
    assert_equal(np.inner(3, arr), desired)