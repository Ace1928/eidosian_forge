import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
def test_hamilton_filter_order_zero_with_tvtp(self):
    k_regimes = 3
    nobs = 8
    initial_probabilities = np.ones(k_regimes) / k_regimes
    regime_transition = np.zeros((k_regimes, k_regimes, nobs))
    regime_transition[...] = np.eye(k_regimes)[:, :, np.newaxis]
    regime_transition[..., 4] = [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]]
    conditional_likelihoods = np.empty((k_regimes, nobs))
    conditional_likelihoods[:, 0] = [1 / 3, 1 / 3, 1 / 3]
    conditional_likelihoods[:, 1] = [1 / 3, 1 / 3, 0]
    conditional_likelihoods[:, 2] = [0, 1 / 3, 1 / 3]
    conditional_likelihoods[:, 3:5] = [[1 / 3], [1 / 3], [1 / 3]]
    conditional_likelihoods[:, 5] = [0, 1 / 3, 1 / 3]
    conditional_likelihoods[:, 6] = [0, 0, 1 / 3]
    conditional_likelihoods[:, 7] = [1 / 3, 1 / 3, 1 / 3]
    expected_marginals = np.empty((k_regimes, nobs))
    expected_marginals[:, 0] = [1 / 3, 1 / 3, 1 / 3]
    expected_marginals[:, 1] = [1 / 2, 1 / 2, 0]
    expected_marginals[:, 2:4] = [[0], [1], [0]]
    expected_marginals[:, 4:6] = [[0], [1 / 2], [1 / 2]]
    expected_marginals[:, 6:8] = [[0], [0], [1]]
    cy_results = markov_switching.cy_hamilton_filter_log(initial_probabilities, regime_transition, np.log(conditional_likelihoods + 1e-20), model_order=0)
    assert_allclose(cy_results[0], expected_marginals, atol=1e-15)