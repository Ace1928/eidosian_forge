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
def test_f5_2_cons_symmetry(self):
    """Symmetry constrained test function"""
    options = {'symmetry': [0, 0], 'disp': True}
    run_test(test1_1, n=200, options=options, iters=1, sampling_method='simplicial')