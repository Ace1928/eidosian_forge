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
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (50, 50), (3, 10, 10)])
@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.complex64, np.complex128))
def test_basic_property(self, shape, dtype):
    np.random.seed(1)
    a = np.random.randn(*shape)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * np.random.randn(*shape)
    t = list(range(len(shape)))
    t[-2:] = (-1, -2)
    a = np.matmul(a.transpose(t).conj(), a)
    a = np.asarray(a, dtype=dtype)
    c = np.linalg.cholesky(a)
    b = np.matmul(c, c.transpose(t).conj())
    with np._no_nep50_warning():
        atol = 500 * a.shape[0] * np.finfo(dtype).eps
    assert_allclose(b, a, atol=atol, err_msg=f'{shape} {dtype}\n{a}\n{c}')