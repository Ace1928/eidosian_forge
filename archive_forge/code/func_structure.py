from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
def structure(n):
    A = np.zeros((n, n), dtype=int)
    A[0, 0] = 1
    A[0, 1] = 1
    for i in range(1, n - 1):
        A[i, i - 1:i + 2] = 1
    A[-1, -1] = 1
    A[-1, -2] = 1
    return A