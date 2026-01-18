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
@pytest.mark.parametrize('method, atol', [('Nelder-Mead', 2e-05), ('COBYLA', 1e-05), ('Powell', 1e-08), ('CG', 1e-08), ('BFGS', 1e-08), ('TNC', 1e-08), ('SLSQP', 2e-07)])
def test_multi_ls_minimizer(self, method, atol):
    ret = dual_annealing(self.func, self.ld_bounds, minimizer_kwargs=dict(method=method), seed=self.seed)
    assert_allclose(ret.fun, 0.0, atol=atol)