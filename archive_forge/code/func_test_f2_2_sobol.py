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
def test_f2_2_sobol(self):
    """Univariate test function on
        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]"""
    run_test(test2_2)