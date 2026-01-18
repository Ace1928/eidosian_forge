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
def test_5_1_simplicial_argless(self):
    """Test Default simplicial sampling settings on TestFunction 1"""
    res = shgo(test1_1.f, test1_1.bounds, constraints=test1_1.cons)
    numpy.testing.assert_allclose(res.x, test1_1.expected_x, rtol=1e-05, atol=1e-05)