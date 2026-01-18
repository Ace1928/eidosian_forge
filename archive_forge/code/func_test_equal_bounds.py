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
def test_equal_bounds(self):
    with np.errstate(invalid='raise'):
        solver = DifferentialEvolutionSolver(self.quadratic, bounds=[(2.0, 2.0), (1.0, 3.0)])
        v = solver._unscale_parameters([2.0, 2.0])
        assert_allclose(v, 0.5)
    res = differential_evolution(self.quadratic, [(2.0, 2.0), (3.0, 3.0)])
    assert_equal(res.x, [2.0, 3.0])