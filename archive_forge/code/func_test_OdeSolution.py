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
def test_OdeSolution():
    ts = np.array([0, 2, 5], dtype=float)
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))
    sol = OdeSolution(ts, [s1, s2])
    assert_equal(sol(-1), [-1])
    assert_equal(sol(1), [-1])
    assert_equal(sol(2), [-1])
    assert_equal(sol(3), [1])
    assert_equal(sol(5), [1])
    assert_equal(sol(6), [1])
    assert_equal(sol([0, 6, -2, 1.5, 4.5, 2.5, 5, 5.5, 2]), np.array([[-1, 1, -1, -1, 1, 1, 1, 1, -1]]))
    ts = np.array([10, 4, -3])
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))
    sol = OdeSolution(ts, [s1, s2])
    assert_equal(sol(11), [-1])
    assert_equal(sol(10), [-1])
    assert_equal(sol(5), [-1])
    assert_equal(sol(4), [-1])
    assert_equal(sol(0), [1])
    assert_equal(sol(-3), [1])
    assert_equal(sol(-4), [1])
    assert_equal(sol([12, -5, 10, -3, 6, 1, 4]), np.array([[-1, 1, -1, 1, -1, 1, -1]]))
    ts = np.array([1, 1])
    s = ConstantDenseOutput(1, 1, np.array([10]))
    sol = OdeSolution(ts, [s])
    assert_equal(sol(0), [10])
    assert_equal(sol(1), [10])
    assert_equal(sol(2), [10])
    assert_equal(sol([2, 1, 0]), np.array([[10, 10, 10]]))