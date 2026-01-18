import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse._sputils import matrix
def check_discrete_case(self, a, q, method=None):
    x = solve_discrete_lyapunov(a, q, method=method)
    assert_array_almost_equal(np.dot(np.dot(a, x), a.conj().transpose()) - x, -1.0 * q)