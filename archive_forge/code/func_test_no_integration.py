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
def test_no_integration():
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        sol = solve_ivp(lambda t, y: -y, [4, 4], [2, 3], method=method, dense_output=True)
        assert_equal(sol.sol(4), [2, 3])
        assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])