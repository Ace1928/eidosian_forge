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
@pytest.mark.skip(reason='Bad memory reports lead to OOM in ci testing')
def test_blas64_dot():
    n = 2 ** 32
    a = np.zeros([1, n], dtype=np.float32)
    b = np.ones([1, 1], dtype=np.float32)
    a[0, -1] = 1
    c = np.dot(b, a)
    assert_equal(c[0, -1], 1)