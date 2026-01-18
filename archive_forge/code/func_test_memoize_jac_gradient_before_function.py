import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
def test_memoize_jac_gradient_before_function(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)
    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function.derivative(x0), 2 * x0)
    assert function_with_gradient.number_of_calls == 1
    assert_allclose(memoized_function(x0), 5.0)
    assert function_with_gradient.number_of_calls == 1, 'function is not recomputed if function value is requested after gradient'
    assert_allclose(memoized_function.derivative(2 * x0), 4 * x0, err_msg='different input triggers new computation')
    assert function_with_gradient.number_of_calls == 2, 'different input triggers new computation'