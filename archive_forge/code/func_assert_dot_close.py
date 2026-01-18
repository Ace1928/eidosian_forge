from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def assert_dot_close(A, X, desired):
    assert_allclose(self.blas_func(1.0, A, X), desired, rtol=1e-05, atol=1e-07)