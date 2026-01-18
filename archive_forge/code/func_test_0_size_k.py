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
def test_0_size_k(self):

    class ArraySubclass(np.ndarray):
        pass
    a = np.arange(4).reshape(1, 2, 2)
    b = np.arange(6).reshape(3, 2, 1).view(ArraySubclass)
    expected = linalg.solve(a, b)[:, :, 0:0]
    result = linalg.solve(a, b[:, :, 0:0])
    assert_array_equal(result, expected)
    assert_(isinstance(result, ArraySubclass))
    expected = linalg.solve(a, b)[:, 0:0, 0:0]
    result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, 0:0])
    assert_array_equal(result, expected)
    assert_(isinstance(result, ArraySubclass))