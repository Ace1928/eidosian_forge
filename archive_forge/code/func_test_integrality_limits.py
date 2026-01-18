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
def test_integrality_limits(self):

    def f(x):
        return x
    integrality = [True, False, True]
    bounds = [(0.2, 1.1), (0.9, 2.2), (3.3, 4.9)]
    solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False, integrality=False)
    assert_allclose(solver.limits[0], [0.2, 0.9, 3.3])
    assert_allclose(solver.limits[1], [1.1, 2.2, 4.9])
    solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False, integrality=integrality)
    assert_allclose(solver.limits[0], [0.5, 0.9, 3.5])
    assert_allclose(solver.limits[1], [1.5, 2.2, 4.5])
    assert_equal(solver.integrality, [True, False, True])
    assert solver.polish is False
    bounds = [(-1.2, -0.9), (0.9, 2.2), (-10.3, 4.1)]
    solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False, integrality=integrality)
    assert_allclose(solver.limits[0], [-1.5, 0.9, -10.5])
    assert_allclose(solver.limits[1], [-0.5, 2.2, 4.5])
    assert_allclose(np.round(solver.limits[0]), [-1.0, 1.0, -10.0])
    assert_allclose(np.round(solver.limits[1]), [-1.0, 2.0, 4.0])
    bounds = [(-10.2, -8.1), (0.9, 2.2), (-10.9, -9.9999)]
    solver = DifferentialEvolutionSolver(f, bounds=bounds, polish=False, integrality=integrality)
    assert_allclose(solver.limits[0], [-10.5, 0.9, -10.5])
    assert_allclose(solver.limits[1], [-8.5, 2.2, -9.5])
    bounds = [(-10.2, -10.1), (0.9, 2.2), (-10.9, -9.9999)]
    with pytest.raises(ValueError, match='One of the integrality'):
        DifferentialEvolutionSolver(f, bounds=bounds, polish=False, integrality=integrality)