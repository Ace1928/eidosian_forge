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
def test_13_high_sobol(self):
    """Test init of high-dimensional sobol sequences"""

    def f(x):
        return 0
    bounds = [(None, None)] * 41
    SHGOc = SHGO(f, bounds, sampling_method='sobol')
    SHGOc.sampling_function(2, 50)