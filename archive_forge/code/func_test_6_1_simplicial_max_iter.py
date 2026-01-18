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
def test_6_1_simplicial_max_iter(self):
    """Test that maximum iteration option works on TestFunction 3"""
    options = {'max_iter': 2}
    res = shgo(test3_1.f, test3_1.bounds, constraints=test3_1.cons, options=options, sampling_method='simplicial')
    numpy.testing.assert_allclose(res.x, test3_1.expected_x, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(res.fun, test3_1.expected_fun, atol=1e-05)