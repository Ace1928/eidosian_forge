import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_get_blas_funcs():
    f1, f2, f3 = get_blas_funcs(('axpy', 'axpy', 'axpy'), (np.empty((2, 2), dtype=np.complex64, order='F'), np.empty((2, 2), dtype=np.complex128, order='C')))
    assert_equal(f1.typecode, 'z')
    assert_equal(f2.typecode, 'z')
    if cblas is not None:
        assert_equal(f1.module_name, 'cblas')
        assert_equal(f2.module_name, 'cblas')
    f1 = get_blas_funcs('rotg')
    assert_equal(f1.typecode, 'd')
    f1 = get_blas_funcs('gemm', dtype=np.complex64)
    assert_equal(f1.typecode, 'c')
    f1 = get_blas_funcs('gemm', dtype='F')
    assert_equal(f1.typecode, 'c')
    f1 = get_blas_funcs('gemm', dtype=np.clongdouble)
    assert_equal(f1.typecode, 'z')
    f1 = get_blas_funcs('axpy', (np.empty((2, 2), dtype=np.float64), np.empty((2, 2), dtype=np.complex64)))
    assert_equal(f1.typecode, 'z')