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
def test_6_1_lower_known_f_min(self):
    """Test Global mode limiting local evaluations with f* too high"""
    options = {'f_min': test2_1.expected_fun + 2.0, 'f_tol': 1e-06, 'minimize_every_iter': True, 'local_iter': 1, 'infty_constraints': False}
    args = (test2_1.f, test2_1.bounds)
    kwargs = {'constraints': test2_1.cons, 'n': None, 'iters': None, 'options': options, 'sampling_method': 'sobol'}
    warns(UserWarning, shgo, *args, **kwargs)