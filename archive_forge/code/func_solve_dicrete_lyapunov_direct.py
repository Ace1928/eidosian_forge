import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def solve_dicrete_lyapunov_direct(self, a, q, complex_step=False):
    if not complex_step:
        lhs = np.kron(a, a.conj())
        lhs = np.eye(lhs.shape[0]) - lhs
        x = np.linalg.solve(lhs, q.flatten())
    else:
        lhs = np.kron(a, a)
        lhs = np.eye(lhs.shape[0]) - lhs
        x = np.linalg.solve(lhs, q.flatten())
    return np.reshape(x, q.shape)