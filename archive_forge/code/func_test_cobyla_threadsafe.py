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
def test_cobyla_threadsafe():
    import concurrent.futures
    import time

    def objective1(x):
        time.sleep(0.1)
        return x[0] ** 2

    def objective2(x):
        time.sleep(0.1)
        return (x[0] - 1) ** 2
    min_method = 'COBYLA'

    def minimizer1():
        return optimize.minimize(objective1, [0.0], method=min_method)

    def minimizer2():
        return optimize.minimize(objective2, [0.0], method=min_method)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = []
        tasks.append(pool.submit(minimizer1))
        tasks.append(pool.submit(minimizer2))
        for t in tasks:
            t.result()