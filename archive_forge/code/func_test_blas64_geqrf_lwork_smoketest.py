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
@pytest.mark.xfail(not HAS_LAPACK64, reason='Numpy not compiled with 64-bit BLAS/LAPACK')
def test_blas64_geqrf_lwork_smoketest():
    dtype = np.float64
    lapack_routine = np.linalg.lapack_lite.dgeqrf
    m = 2 ** 32 + 1
    n = 2 ** 32 + 1
    lda = m
    a = np.zeros([1, 1], dtype=dtype)
    work = np.zeros([1], dtype=dtype)
    tau = np.zeros([1], dtype=dtype)
    results = lapack_routine(m, n, a, lda, tau, work, -1, 0)
    assert_equal(results['info'], 0)
    assert_equal(results['m'], m)
    assert_equal(results['n'], m)
    lwork = int(work.item())
    assert_(2 ** 32 < lwork < 2 ** 42)