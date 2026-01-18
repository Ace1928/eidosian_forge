from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
def matrixmultiply(a, b):
    if len(b.shape) == 1:
        b_is_vector = True
        b = b[:, newaxis]
    else:
        b_is_vector = False
    assert_(a.shape[1] == b.shape[0])
    c = zeros((a.shape[0], b.shape[1]), common_type(a, b))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            s = 0
            for k in range(a.shape[1]):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    if b_is_vector:
        c = c.reshape((a.shape[0],))
    return c