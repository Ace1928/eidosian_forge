import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_7_3_minkwargs(self):
    """Test minimizer_kwargs arguments for solvers without constraints"""
    for solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:

        def jac(x):
            return numpy.array([2 * x[0], 2 * x[1]]).T

        def hess(x):
            return numpy.array([[2, 0], [0, 2]])
        minimizer_kwargs = {'method': solver, 'jac': jac, 'hess': hess}
        logging.info(f'Solver = {solver}')
        logging.info('=' * 100)
        run_test(test1_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')