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
def test_invalid_mutation_values_arent_accepted(self):
    func = rosen
    mutation = (0.5, 3)
    assert_raises(ValueError, DifferentialEvolutionSolver, func, self.bounds, mutation=mutation)
    mutation = (-1, 1)
    assert_raises(ValueError, DifferentialEvolutionSolver, func, self.bounds, mutation=mutation)
    mutation = (0.1, np.nan)
    assert_raises(ValueError, DifferentialEvolutionSolver, func, self.bounds, mutation=mutation)
    mutation = 0.5
    solver = DifferentialEvolutionSolver(func, self.bounds, mutation=mutation)
    assert_equal(0.5, solver.scale)
    assert_equal(None, solver.dither)