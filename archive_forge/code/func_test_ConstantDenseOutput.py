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
def test_ConstantDenseOutput():
    sol = ConstantDenseOutput(0, 1, np.array([1, 2]))
    assert_allclose(sol(1.5), [1, 2])
    assert_allclose(sol([1, 1.5, 2]), [[1, 1, 1], [2, 2, 2]])
    sol = ConstantDenseOutput(0, 1, np.array([]))
    assert_allclose(sol(1.5), np.empty(0))
    assert_allclose(sol([1, 1.5, 2]), np.empty((0, 3)))