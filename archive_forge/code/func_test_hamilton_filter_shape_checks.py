import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
def test_hamilton_filter_shape_checks(self):
    k_regimes = 3
    nobs = 8
    order = 3
    initial_probabilities = np.ones(k_regimes) / k_regimes
    regime_transition = np.ones((k_regimes, k_regimes, nobs)) / k_regimes
    conditional_loglikelihoods = np.ones(order * (k_regimes,) + (nobs,))
    with assert_raises(ValueError):
        markov_switching.cy_hamilton_filter_log(initial_probabilities, regime_transition, conditional_loglikelihoods, model_order=order)