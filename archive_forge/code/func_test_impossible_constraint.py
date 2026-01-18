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
def test_impossible_constraint(self):

    def constr_f(x):
        return np.array([x[0] + x[1]])
    nlc = NonlinearConstraint(constr_f, -np.inf, -1)
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=nlc, popsize=3, seed=1)
    with warns(UserWarning):
        res = solver.solve()
    assert res.maxcv > 0
    assert not res.success
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=nlc, polish=False)
    next(solver)
    assert not solver.feasible.all()
    assert not np.isfinite(solver.population_energies).all()
    l = 20
    cv = solver.constraint_violation[0]
    solver.population_energies[[0, l]] = solver.population_energies[[l, 0]]
    solver.population[[0, l], :] = solver.population[[l, 0], :]
    solver.constraint_violation[[0, l], :] = solver.constraint_violation[[l, 0], :]
    solver._promote_lowest_energy()
    assert_equal(solver.constraint_violation[0], cv)