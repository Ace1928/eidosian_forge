import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def test_dynamic_programming_logic(self):
    arrays = [np.random.random((30, 35)), np.random.random((35, 15)), np.random.random((15, 5)), np.random.random((5, 10)), np.random.random((10, 20)), np.random.random((20, 25))]
    m_expected = np.array([[0.0, 15750.0, 7875.0, 9375.0, 11875.0, 15125.0], [0.0, 0.0, 2625.0, 4375.0, 7125.0, 10500.0], [0.0, 0.0, 0.0, 750.0, 2500.0, 5375.0], [0.0, 0.0, 0.0, 0.0, 1000.0, 3500.0], [0.0, 0.0, 0.0, 0.0, 0.0, 5000.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_expected = np.array([[0, 1, 1, 3, 3, 3], [0, 0, 2, 3, 3, 3], [0, 0, 0, 3, 3, 3], [0, 0, 0, 0, 4, 5], [0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0]], dtype=int)
    s_expected -= 1
    s, m = _multi_dot_matrix_chain_order(arrays, return_costs=True)
    assert_almost_equal(np.triu(s[:-1, 1:]), np.triu(s_expected[:-1, 1:]))
    assert_almost_equal(np.triu(m), np.triu(m_expected))