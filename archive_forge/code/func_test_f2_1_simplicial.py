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
def test_f2_1_simplicial(self):
    """Univariate test function on
        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]"""
    options = {'minimize_every_iter': False}
    run_test(test2_1, n=200, iters=7, options=options, sampling_method='simplicial')