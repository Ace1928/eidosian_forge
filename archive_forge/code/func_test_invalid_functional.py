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
def test_invalid_functional(self):

    def func(x):
        return np.array([np.sum(x ** 2), np.sum(x)])
    with assert_raises(RuntimeError, match='func\\(x, \\*args\\) must return a scalar value'):
        differential_evolution(func, [(-2, 2), (-2, 2)])