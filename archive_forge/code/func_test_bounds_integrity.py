from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
def test_bounds_integrity(self):
    wrong_bounds = [(-5.12, 5.12), (1, 0), (5.12, 5.12)]
    assert_raises(ValueError, dual_annealing, self.func, wrong_bounds)