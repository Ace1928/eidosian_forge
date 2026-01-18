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
def test_can_init_with_dithering(self):
    mutation = (0.5, 1)
    solver = DifferentialEvolutionSolver(self.quadratic, self.bounds, mutation=mutation)
    assert_equal(solver.dither, list(mutation))