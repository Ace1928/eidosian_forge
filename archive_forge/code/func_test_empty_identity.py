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
def test_empty_identity(self):
    """ Empty input should put an identity matrix in u or vh """
    x = np.empty((4, 0))
    u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
    assert_equal(u.shape, (4, 4))
    assert_equal(vh.shape, (0, 0))
    assert_equal(u, np.eye(4))
    x = np.empty((0, 4))
    u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
    assert_equal(u.shape, (0, 0))
    assert_equal(vh.shape, (4, 4))
    assert_equal(vh, np.eye(4))