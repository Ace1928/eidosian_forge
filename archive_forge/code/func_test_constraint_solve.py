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
def test_constraint_solve(self):

    def constr_f(x):
        return np.array([x[0] + x[1]])
    nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)
    solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)], constraints=nlc)
    with warns(UserWarning):
        res = solver.solve()
    assert constr_f(res.x) <= 1.9
    assert res.success