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
def test__mutate1(self):
    result = np.array([0.05])
    trial = self.dummy_solver2._best1((2, 3, 4, 5, 6))
    assert_allclose(trial, result)
    result = np.array([0.25])
    trial = self.dummy_solver2._rand1((2, 3, 4, 5, 6))
    assert_allclose(trial, result)