import pickle
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
import pytest
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .results import results_kalman_filter
def test_kalman_filter_pickle(data):
    true = results_kalman_filter.uc_uni
    k_states = 4
    model = KalmanFilter(k_endog=1, k_states=k_states)
    model.bind(data['lgdp'].values)
    model.design[:, :, 0] = [1, 1, 0, 0]
    model.transition[[0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0]] = [1, 1, 0, 0, 1, 1]
    model.selection = np.eye(model.k_states)
    sigma_v, sigma_e, sigma_w, phi_1, phi_2 = np.array(true['parameters'])
    model.transition[[1, 1], [1, 2], [0, 0]] = [phi_1, phi_2]
    model.state_cov[np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [sigma_v ** 2, sigma_e ** 2, 0, sigma_w ** 2]
    initial_state = np.zeros((k_states,))
    initial_state_cov = np.eye(k_states) * 100
    initial_state_cov = np.dot(np.dot(model.transition[:, :, 0], initial_state_cov), model.transition[:, :, 0].T)
    model.initialize_known(initial_state, initial_state_cov)
    pkl_mod = pickle.loads(pickle.dumps(model))
    results = model.filter()
    pkl_results = pkl_mod.filter()
    assert_allclose(results.llf_obs[true['start']:].sum(), pkl_results.llf_obs[true['start']:].sum())
    assert_allclose(results.filtered_state[0][true['start']:], pkl_results.filtered_state[0][true['start']:])
    assert_allclose(results.filtered_state[1][true['start']:], pkl_results.filtered_state[1][true['start']:])
    assert_allclose(results.filtered_state[3][true['start']:], pkl_results.filtered_state[3][true['start']:])