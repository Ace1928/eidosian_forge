import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
def test_impulse_responses():
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    mod['design', 0, 0] = 1.0
    mod['transition', 0, 0] = 1.0
    mod['selection', 0, 0] = 1.0
    mod['state_cov', 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10)
    desired = np.ones((11, 1))
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    mod['design', 0, 0] = 1.0
    mod['transition', 0, 0] = 1.0
    mod['selection', 0, 0] = 1.0
    mod['state_cov', 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10, impulse=[2])
    desired = np.ones((11, 1)) * 2
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    mod['design', 0, 0] = 1.0
    mod['transition', 0, 0] = 1.0
    mod['selection', 0, 0] = 1.0
    mod['state_cov', 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10, orthogonalized=True)
    desired = np.ones((11, 1)) * 2 ** 0.5
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    mod['design', 0, 0] = 1.0
    mod['transition', 0, 0] = 1.0
    mod['selection', 0, 0] = 1.0
    mod['state_cov', 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10, orthogonalized=True, cumulative=True)
    desired = np.cumsum(np.ones((11, 1)) * 2 ** 0.5)[:, np.newaxis]
    actual = mod.impulse_responses(steps=10, impulse=[1], orthogonalized=True, cumulative=True)
    desired = np.cumsum(np.ones((11, 1)) * 2 ** 0.5)[:, np.newaxis]
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    mod['state_intercept', 0] = 100.0
    mod['design', 0, 0] = 1.0
    mod['obs_intercept', 0] = -1000.0
    mod['transition', 0, 0] = 1.0
    mod['selection', 0, 0] = 1.0
    mod['state_cov', 0, 0] = 2.0
    actual = mod.impulse_responses(steps=10)
    desired = np.ones((11, 1))
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=1, initialization='diffuse')
    assert_raises(ValueError, mod.impulse_responses, impulse=1)
    assert_raises(ValueError, mod.impulse_responses, impulse=[1, 1])
    assert_raises(ValueError, mod.impulse_responses, impulse=[])
    mod = SimulationSmoother(k_endog=1, k_states=2, initialization='diffuse')
    mod['design', 0, 0:2] = 1.0
    mod['transition', :, :] = np.eye(2)
    mod['selection', :, :] = np.eye(2)
    mod['state_cov', :, :] = np.eye(2)
    desired = np.ones((11, 1))
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[1, 0])
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[0, 1])
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[1, 0], orthogonalized=True)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=[0, 1], orthogonalized=True)
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=1, k_states=2, initialization='diffuse')
    mod['design', 0, 0:2] = 1.0
    mod['transition', :, :] = np.eye(2)
    mod['selection', :, :] = np.eye(2)
    mod['state_cov', :, :] = np.array([[1, 0.5], [0.5, 1.25]])
    desired = np.ones((11, 1))
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, desired)
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, desired + desired * 0.5)
    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, desired)
    mod = SimulationSmoother(k_endog=2, k_states=2, initialization='diffuse')
    mod['design', :, :] = np.eye(2)
    mod['transition', :, :] = np.eye(2)
    mod['selection', :, :] = np.eye(2)
    mod['state_cov', :, :] = np.array([[1, 0.5], [0.5, 1.25]])
    ones = np.ones((11, 1))
    zeros = np.zeros((11, 1))
    actual = mod.impulse_responses(steps=10, impulse=0)
    assert_allclose(actual, np.c_[ones, zeros])
    actual = mod.impulse_responses(steps=10, impulse=1)
    assert_allclose(actual, np.c_[zeros, ones])
    actual = mod.impulse_responses(steps=10, impulse=0, orthogonalized=True)
    assert_allclose(actual, np.c_[ones, ones * 0.5])
    actual = mod.impulse_responses(steps=10, impulse=1, orthogonalized=True)
    assert_allclose(actual, np.c_[zeros, ones])
    mod = sarimax.SARIMAX([0.1, 0.5, -0.2], order=(1, 0, 0))
    phi = 0.5
    mod.update([phi, 1])
    desired = np.cumprod(np.r_[1, [phi] * 10])
    actual = mod.ssm.impulse_responses(steps=10)
    assert_allclose(actual[:, 0], desired)
    res = mod.filter([phi, 1.0])
    actual = res.impulse_responses(steps=10)
    assert_allclose(actual, desired)