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
@pytest.mark.parametrize(['c1', 'c2'], [[0.5, 2], [-0.1, 0.1], [0.2, 0.1]])
def test_invalid_c1_c2(self, c1, c2):
    with pytest.raises(ValueError, match="'c1' and 'c2'"):
        x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
        optimize.minimize(optimize.rosen, x0, method='cg', options={'c1': c1, 'c2': c2})