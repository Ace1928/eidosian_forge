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
def test_bounds_checking(self):
    func = rosen
    bounds = [-3]
    assert_raises(ValueError, differential_evolution, func, bounds)
    bounds = [(-3, 3), (3, 4, 5)]
    assert_raises(ValueError, differential_evolution, func, bounds)
    result = differential_evolution(rosen, Bounds([0, 0], [2, 2]))
    assert_almost_equal(result.x, (1.0, 1.0))