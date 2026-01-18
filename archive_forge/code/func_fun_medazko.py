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
def fun_medazko(t, y):
    n = y.shape[0] // 2
    k = 100
    c = 4
    phi = 2 if t <= 5 else 0
    y = np.hstack((phi, 0, y, y[-2]))
    d = 1 / n
    j = np.arange(n) + 1
    alpha = 2 * (j * d - 1) ** 3 / c ** 2
    beta = (j * d - 1) ** 4 / c ** 2
    j_2_p1 = 2 * j + 2
    j_2_m3 = 2 * j - 2
    j_2_m1 = 2 * j
    j_2 = 2 * j + 1
    f = np.empty(2 * n)
    f[::2] = alpha * (y[j_2_p1] - y[j_2_m3]) / (2 * d) + beta * (y[j_2_m3] - 2 * y[j_2_m1] + y[j_2_p1]) / d ** 2 - k * y[j_2_m1] * y[j_2]
    f[1::2] = -k * y[j_2] * y[j_2_m1]
    return f