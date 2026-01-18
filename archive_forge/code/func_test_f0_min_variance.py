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
@pytest.mark.skip('Not a test')
def test_f0_min_variance(self):
    """Return a minimum on a perfectly symmetric problem, based on
            gh10429"""
    avg = 0.5
    cons = {'type': 'eq', 'fun': lambda x: numpy.mean(x) - avg}
    res = shgo(numpy.var, bounds=6 * [(0, 1)], constraints=cons)
    assert res.success
    assert_allclose(res.fun, 0, atol=1e-15)
    assert_allclose(res.x, 0.5)