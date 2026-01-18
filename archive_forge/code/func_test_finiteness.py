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
def test_finiteness():
    nm = np.full((2, 2), np.nan)
    sq = np.eye(2)
    for x in (solve_continuous_are, solve_discrete_are):
        assert_raises(ValueError, x, nm, sq, sq, sq)
        assert_raises(ValueError, x, sq, nm, sq, sq)
        assert_raises(ValueError, x, sq, sq, nm, sq)
        assert_raises(ValueError, x, sq, sq, sq, nm)
        assert_raises(ValueError, x, sq, sq, sq, sq, nm)
        assert_raises(ValueError, x, sq, sq, sq, sq, sq, nm)