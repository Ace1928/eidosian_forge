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
def test_memoize_jac_with_bfgs(function_with_gradient):
    """ Tests that using MemoizedJac in combination with ScalarFunction
        and BFGS does not lead to repeated function evaluations.
        Tests changes made in response to GH11868.
    """
    memoized_function = MemoizeJac(function_with_gradient)
    jac = memoized_function.derivative
    hess = optimize.BFGS()
    x0 = np.array([1.0, 0.5])
    scalar_function = ScalarFunction(memoized_function, x0, (), jac, hess, None, None)
    assert function_with_gradient.number_of_calls == 1
    scalar_function.fun(x0 + 0.1)
    assert function_with_gradient.number_of_calls == 2
    scalar_function.fun(x0 + 0.2)
    assert function_with_gradient.number_of_calls == 3