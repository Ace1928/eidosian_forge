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
def check_stationary_initialization_2dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)
    mod = MLEModel(endog, k_states=2, k_posdef=2)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3, -10.2], dtype=dtype)
    phi = np.array([[0.8, 0.1], [-0.2, 0.7]], dtype=dtype)
    sigma2 = np.array([[1.4, -0.2], [-0.2, 4.5]], dtype=dtype)
    mod['state_intercept'] = intercept
    mod['transition'] = phi
    mod['selection'] = np.eye(2).astype(dtype)
    mod['state_cov'] = sigma2
    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()
    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)
    desired = np.linalg.solve(np.eye(2).astype(dtype) - phi, intercept)
    assert_allclose(initial_state, desired)
    desired = solve_discrete_lyapunov(phi, sigma2)
    assert_allclose(initial_state_cov, desired, atol=1e-05)