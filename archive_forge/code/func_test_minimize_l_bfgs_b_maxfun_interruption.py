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
def test_minimize_l_bfgs_b_maxfun_interruption(self):
    f = optimize.rosen
    g = optimize.rosen_der
    values = []
    x0 = np.full(7, 1000)

    def objfun(x):
        value = f(x)
        values.append(value)
        return value
    low, medium, high = (30, 100, 300)
    optimize.fmin_l_bfgs_b(objfun, x0, fprime=g, maxfun=high)
    v, k = max(((y, i) for i, y in enumerate(values[medium:])))
    maxfun = medium + k
    target = min(values[:low])
    xmin, fmin, d = optimize.fmin_l_bfgs_b(f, x0, fprime=g, maxfun=maxfun)
    assert_array_less(fmin, target)