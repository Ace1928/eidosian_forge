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
def test_calculate_population_energies(self):
    solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=3)
    solver._calculate_population_energies(solver.population)
    solver._promote_lowest_energy()
    assert_equal(np.argmin(solver.population_energies), 0)
    assert_equal(solver._nfev, 6)