import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_absolute_step_sign():

    def f(x):
        return -np.abs(x[0] + 1) + np.abs(x[1] + 1)
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=1e-08)
    assert_allclose(grad, [-1.0, 1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=-1e-08)
    assert_allclose(grad, [1.0, -1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=[1e-08, 1e-08])
    assert_allclose(grad, [-1.0, 1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=[1e-08, -1e-08])
    assert_allclose(grad, [-1.0, -1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=[-1e-08, 1e-08])
    assert_allclose(grad, [1.0, 1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=1e-08, bounds=(-np.inf, -1))
    assert_allclose(grad, [1.0, -1.0])
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=-1e-08, bounds=(-1, np.inf))
    assert_allclose(grad, [-1.0, 1.0])