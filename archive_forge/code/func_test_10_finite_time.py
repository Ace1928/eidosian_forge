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
@pytest.mark.xfail(IS_PYPY and sys.platform == 'win32', reason='Failing and fix in PyPy not planned (see gh-18632)')
def test_10_finite_time(self):
    """Test single function constraint passing"""
    options = {'maxtime': 1e-15}

    def f(x):
        time.sleep(1e-14)
        return 0.0
    res = shgo(f, test1_1.bounds, iters=5, options=options)
    assert res.nit == 1