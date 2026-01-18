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
def test_maxfun_stops_solve(self):
    solver = DifferentialEvolutionSolver(rosen, self.bounds, maxfun=1, polish=False)
    result = solver.solve()
    assert_equal(result.nfev, 2)
    assert_equal(result.success, False)
    assert_equal(result.message, 'Maximum number of function evaluations has been exceeded.')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=5, polish=False, maxfun=40)
    result = solver.solve()
    assert_equal(result.nfev, 41)
    assert_equal(result.success, False)
    assert_equal(result.message, 'Maximum number of function evaluations has been exceeded.')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=5, polish=False, maxfun=47, updating='deferred')
    result = solver.solve()
    assert_equal(result.nfev, 47)
    assert_equal(result.success, False)
    assert_equal(result.message, 'Maximum number of function evaluations has been reached.')