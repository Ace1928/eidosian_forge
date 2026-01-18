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
def test_array_rtol():

    def f(t, y):
        return (y[0], y[1])
    sol = solve_ivp(f, (0, 1), [1.0, 1.0], rtol=[0.1, 0.1])
    err1 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))
    with pytest.warns(UserWarning, match='At least one element...'):
        sol = solve_ivp(f, (0, 1), [1.0, 1.0], rtol=[0.1, 1e-16])
        err2 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))
    assert err2 < err1