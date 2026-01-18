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
def test_args_tuple_is_passed(self):
    bounds = [(-10, 10)]
    args = (1.0, 2.0, 3.0)

    def quadratic(x, *args):
        if type(args) != tuple:
            raise ValueError('args should be a tuple')
        return args[0] + args[1] * x + args[2] * x ** 2.0
    result = differential_evolution(quadratic, bounds, args=args, polish=True)
    assert_almost_equal(result.fun, 2 / 3.0)