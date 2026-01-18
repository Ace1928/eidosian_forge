from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
def lowerBidiagonalMatrix(m, n):
    if m <= n:
        row = hstack((arange(m, dtype=int), arange(1, m, dtype=int)))
        col = hstack((arange(m, dtype=int), arange(m - 1, dtype=int)))
        data = hstack((arange(1, m + 1, dtype=float), arange(1, m, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m, n))
    else:
        row = hstack((arange(n, dtype=int), arange(1, n + 1, dtype=int)))
        col = hstack((arange(n, dtype=int), arange(n, dtype=int)))
        data = hstack((arange(1, n + 1, dtype=float), arange(1, n + 1, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m, n))