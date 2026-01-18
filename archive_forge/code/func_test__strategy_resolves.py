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
def test__strategy_resolves(self):
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='best1exp')
    assert_equal(solver.strategy, 'best1exp')
    assert_equal(solver.mutation_func.__name__, '_best1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='best1bin')
    assert_equal(solver.strategy, 'best1bin')
    assert_equal(solver.mutation_func.__name__, '_best1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='rand1bin')
    assert_equal(solver.strategy, 'rand1bin')
    assert_equal(solver.mutation_func.__name__, '_rand1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='rand1exp')
    assert_equal(solver.strategy, 'rand1exp')
    assert_equal(solver.mutation_func.__name__, '_rand1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='rand2exp')
    assert_equal(solver.strategy, 'rand2exp')
    assert_equal(solver.mutation_func.__name__, '_rand2')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='best2bin')
    assert_equal(solver.strategy, 'best2bin')
    assert_equal(solver.mutation_func.__name__, '_best2')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='rand2bin')
    assert_equal(solver.strategy, 'rand2bin')
    assert_equal(solver.mutation_func.__name__, '_rand2')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='rand2exp')
    assert_equal(solver.strategy, 'rand2exp')
    assert_equal(solver.mutation_func.__name__, '_rand2')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='randtobest1bin')
    assert_equal(solver.strategy, 'randtobest1bin')
    assert_equal(solver.mutation_func.__name__, '_randtobest1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='randtobest1exp')
    assert_equal(solver.strategy, 'randtobest1exp')
    assert_equal(solver.mutation_func.__name__, '_randtobest1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='currenttobest1bin')
    assert_equal(solver.strategy, 'currenttobest1bin')
    assert_equal(solver.mutation_func.__name__, '_currenttobest1')
    solver = DifferentialEvolutionSolver(rosen, self.bounds, strategy='currenttobest1exp')
    assert_equal(solver.strategy, 'currenttobest1exp')
    assert_equal(solver.mutation_func.__name__, '_currenttobest1')