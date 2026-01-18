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
def test_1_maxiter(self):
    """Test failure on insufficient iterations"""
    options = {'maxiter': 2}
    res = shgo(test4_1.f, test4_1.bounds, n=2, iters=None, options=options, sampling_method='sobol')
    numpy.testing.assert_equal(False, res.success)
    numpy.testing.assert_equal(4, res.tnev)