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
def test_complex_high_ord(self):
    d = np.empty((2,), dtype=np.clongdouble)
    d[0] = 6 + 7j
    d[1] = -6 + 7j
    res = 11.615898132184
    old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=10)
    d = d.astype(np.complex128)
    old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=9)
    d = d.astype(np.complex64)
    old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=5)