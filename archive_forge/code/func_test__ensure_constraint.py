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
def test__ensure_constraint(self):
    trial = np.array([1.1, -100, 0.9, 2.0, 300.0, -1e-05])
    self.dummy_solver._ensure_constraint(trial)
    assert_equal(trial[2], 0.9)
    assert_(np.logical_and(trial >= 0, trial <= 1).all())