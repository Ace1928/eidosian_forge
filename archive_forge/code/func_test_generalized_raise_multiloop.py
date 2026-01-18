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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_generalized_raise_multiloop():
    invertible = np.array([[1, 2], [3, 4]])
    non_invertible = np.array([[1, 1], [1, 1]])
    x = np.zeros([4, 4, 2, 2])[1::2]
    x[...] = invertible
    x[0, 0] = non_invertible
    assert_raises(np.linalg.LinAlgError, np.linalg.inv, x)