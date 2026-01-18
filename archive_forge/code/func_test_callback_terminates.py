import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_callback_terminates(self):
    bounds = [(0, 2), (0, 2)]
    expected_msg = 'callback function requested stop early'

    def callback_python_true(param, convergence=0.0):
        return True
    result = differential_evolution(rosen, bounds, callback=callback_python_true)
    assert_string_equal(result.message, expected_msg)

    def callback_stop(intermediate_result):
        raise StopIteration
    result = differential_evolution(rosen, bounds, callback=callback_stop)
    assert not result.success

    def callback_evaluates_true(param, convergence=0.0):
        return [10]
    result = differential_evolution(rosen, bounds, callback=callback_evaluates_true)
    assert_string_equal(result.message, expected_msg)
    assert not result.success

    def callback_evaluates_false(param, convergence=0.0):
        return []
    result = differential_evolution(rosen, bounds, callback=callback_evaluates_false)
    assert result.success