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
@pytest.mark.slow
def test_f4_simplicial(self):
    """NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)"""
    run_test(test4_1, n=1, sampling_method='simplicial')