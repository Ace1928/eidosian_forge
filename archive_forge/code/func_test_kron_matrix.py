import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_kron_matrix():
    a = np.ones([2, 2])
    m = np.asmatrix(a)
    assert_equal(type(np.kron(a, a)), np.ndarray)
    assert_equal(type(np.kron(m, m)), np.matrix)
    assert_equal(type(np.kron(a, m)), np.matrix)
    assert_equal(type(np.kron(m, a)), np.matrix)