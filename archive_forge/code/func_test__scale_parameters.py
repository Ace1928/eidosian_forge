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
def test__scale_parameters(self):
    trial = np.array([0.3])
    assert_equal(30, self.dummy_solver._scale_parameters(trial))
    self.dummy_solver.limits = np.array([[100], [0.0]])
    assert_equal(30, self.dummy_solver._scale_parameters(trial))