import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_al_mohy_higham_2012_experiment_1(self):
    A = _get_al_mohy_higham_2012_experiment_1()
    A_funm_sqrt, info = funm(A, np.sqrt, disp=False)
    A_sqrtm, info = sqrtm(A, disp=False)
    A_rem_power = _matfuncs_inv_ssq._remainder_matrix_power(A, 0.5)
    A_power = fractional_matrix_power(A, 0.5)
    assert_allclose(A_rem_power, A_power, rtol=1e-11)
    assert_allclose(A_sqrtm, A_power)
    assert_allclose(A_sqrtm, A_funm_sqrt)
    for p in (1 / 2, 5 / 3):
        A_power = fractional_matrix_power(A, p)
        A_round_trip = fractional_matrix_power(A_power, 1 / p)
        assert_allclose(A_round_trip, A, rtol=0.01)
        assert_allclose(np.tril(A_round_trip, 1), np.tril(A, 1))