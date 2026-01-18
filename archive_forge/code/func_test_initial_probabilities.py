import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def test_initial_probabilities():
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0.5, 0.5, 1.0]
    mod.initialize_known([0.2, 0.8])
    assert_allclose(mod.initial_probabilities(params), [0.2, 0.8])
    assert_raises(ValueError, mod.initialize_known, [0.2, 0.2, 0.6])
    assert_raises(ValueError, mod.initialize_known, [0.2, 0.2])
    mod.initialize_steady_state()
    assert_allclose(mod.initial_probabilities(params), [0.5, 0.5])
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=endog)
    assert_raises(ValueError, mod.initialize_steady_state)