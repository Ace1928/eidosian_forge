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
def test_gradient_gnev(self):
    minimizer_opts = {'jac': self.rosen_der_wrapper}
    ret = dual_annealing(rosen, self.ld_bounds, minimizer_kwargs=minimizer_opts, seed=self.seed)
    assert ret.njev == self.ngev