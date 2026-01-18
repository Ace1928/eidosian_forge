import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def check_transition_3(params):
    assert_equal(params['regime_transition'], np.s_[0:6])
    assert_equal(params[0, 'regime_transition'], [0, 3])
    assert_equal(params[1, 'regime_transition'], [1, 4])
    assert_equal(params[2, 'regime_transition'], [2, 5])
    assert_equal(params['regime_transition', 0], [0, 3])
    assert_equal(params['regime_transition', 1], [1, 4])
    assert_equal(params['regime_transition', 2], [2, 5])