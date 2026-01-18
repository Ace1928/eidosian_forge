import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
@pytest.mark.smoke
def test_avoid_underflow():
    m = markov_regression.MarkovRegression(gh5380_series, k_regimes=2, switching_variance=True)
    params = np.array([0.697337611, 0.626116329, -6.41266551e-06, 3.81141202e-06, 4.72462327e-08, 4.45291473e-06])
    res = m.smooth(params)
    assert not np.any(np.isnan(res.predicted_joint_probabilities))
    assert not np.any(np.isnan(res.filtered_joint_probabilities))
    assert not np.any(np.isnan(res.smoothed_joint_probabilities))