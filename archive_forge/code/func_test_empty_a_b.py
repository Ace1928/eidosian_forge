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
@pytest.mark.parametrize(['m', 'n', 'n_rhs'], [(4, 2, 2), (0, 4, 1), (0, 4, 2), (4, 0, 1), (4, 0, 2), (4, 2, 0), (0, 0, 0)])
def test_empty_a_b(self, m, n, n_rhs):
    a = np.arange(m * n).reshape(m, n)
    b = np.ones((m, n_rhs))
    x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
    if m == 0:
        assert_((x == 0).all())
    assert_equal(x.shape, (n, n_rhs))
    assert_equal(residuals.shape, (n_rhs,) if m > n else (0,))
    if m > n and n_rhs > 0:
        r = b - np.dot(a, x)
        assert_almost_equal(residuals, (r * r).sum(axis=-2))
    assert_equal(rank, min(m, n))
    assert_equal(s.shape, (min(m, n),))