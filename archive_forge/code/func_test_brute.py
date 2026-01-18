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
def test_brute(self):
    resbrute = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.fmin)
    assert_allclose(resbrute[0], self.solution, atol=0.001)
    assert_allclose(resbrute[1], brute_func(self.solution, *self.params), atol=0.001)
    resbrute = optimize.brute(brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.minimize)
    assert_allclose(resbrute[0], self.solution, atol=0.001)
    assert_allclose(resbrute[1], brute_func(self.solution, *self.params), atol=0.001)
    resbrute = optimize.brute(self.brute_func, self.rranges, args=self.params, full_output=True, finish=optimize.minimize)
    assert_allclose(resbrute[0], self.solution, atol=0.001)