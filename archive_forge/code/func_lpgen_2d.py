import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def lpgen_2d(m, n):
    """ -> A b c LP test: m*n vars, m+n constraints
        row sums == n/m, col sums == 1
        https://gist.github.com/denis-bz/8647461
    """
    np.random.seed(0)
    c = -np.random.exponential(size=(m, n))
    Arow = np.zeros((m, m * n))
    brow = np.zeros(m)
    for j in range(m):
        j1 = j + 1
        Arow[j, j * n:j1 * n] = 1
        brow[j] = n / m
    Acol = np.zeros((n, m * n))
    bcol = np.zeros(n)
    for j in range(n):
        j1 = j + 1
        Acol[j, j::n] = 1
        bcol[j] = 1
    A = np.vstack((Arow, Acol))
    b = np.hstack((brow, bcol))
    return (A, b, c.ravel())