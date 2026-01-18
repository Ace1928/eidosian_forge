import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_copy_missing_diagonal_submatrix():
    nobs = 5
    k_endog = 3
    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1
    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :n, t] = np.eye(n)
    B = np.zeros((k_endog, k_endog, nobs), order='F')
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, True, False, inplace=True)
    assert_equal(B, A)
    B = np.zeros((k_endog, k_endog, nobs), order='F')
    tools.copy_missing_matrix(A, B, missing, True, True, True, inplace=True)
    assert_equal(B, A)