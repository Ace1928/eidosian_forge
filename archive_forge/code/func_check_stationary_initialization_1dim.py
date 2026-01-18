import copy
import pickle
import numpy as np
import pandas as pd
import os
import pytest
from scipy.linalg.blas import find_best_blas_type
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace import _representation, _kalman_filter
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
def check_stationary_initialization_1dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)
    mod = MLEModel(endog, k_states=1, k_posdef=1)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3], dtype=dtype)
    phi = np.diag([0.9]).astype(dtype)
    sigma2 = np.diag([1.3]).astype(dtype)
    mod['state_intercept'] = intercept
    mod['transition'] = phi
    mod['selection'] = np.eye(1).astype(dtype)
    mod['state_cov'] = sigma2
    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()
    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)
    assert_allclose(initial_state, intercept / (1 - phi[0, 0]))
    desired = np.linalg.inv(np.eye(1) - phi).dot(intercept)
    assert_allclose(initial_state, desired)
    assert_allclose(initial_state_cov, sigma2 / (1 - phi ** 2))
    assert_allclose(initial_state_cov, solve_discrete_lyapunov(phi, sigma2))