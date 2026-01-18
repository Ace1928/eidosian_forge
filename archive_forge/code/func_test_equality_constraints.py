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
def test_equality_constraints():
    bounds = [(0.9, 4.0)] * 2

    def faulty(x):
        return x[0] + x[1]
    nlc = NonlinearConstraint(faulty, 3.9, 3.9)
    res = shgo(rosen, bounds=bounds, constraints=nlc)
    assert_allclose(np.sum(res.x), 3.9)

    def faulty(x):
        return x[0] + x[1] - 3.9
    constraints = {'type': 'eq', 'fun': faulty}
    res = shgo(rosen, bounds=bounds, constraints=constraints)
    assert_allclose(np.sum(res.x), 3.9)
    bounds = [(0, 1.0)] * 4

    def faulty(x):
        return x[0] + x[1] + x[2] + x[3] - 1
    constraints = {'type': 'eq', 'fun': faulty}
    res = shgo(lambda x: -np.prod(x), bounds=bounds, constraints=constraints, sampling_method='sobol')
    assert_allclose(np.sum(res.x), 1.0)