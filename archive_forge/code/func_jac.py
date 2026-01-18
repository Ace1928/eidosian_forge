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
def jac(t, y):
    return np.array([[-0.04, 10000.0 * y[2], 10000.0 * y[1]], [0.04, -10000.0 * y[2] - 60000000.0 * y[1], -10000.0 * y[1]], [0, 60000000.0 * y[1], 0]])