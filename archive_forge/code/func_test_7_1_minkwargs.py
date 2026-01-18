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
def test_7_1_minkwargs(self):
    """Test the minimizer_kwargs arguments for solvers with constraints"""
    for solver in ['COBYLA', 'SLSQP']:
        minimizer_kwargs = {'method': solver, 'constraints': test3_1.cons}
        run_test(test3_1, n=100, test_atol=0.001, minimizer_kwargs=minimizer_kwargs, sampling_method='sobol')