import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
def test_dfm(missing=None):
    mod = dynamic_factor.DynamicFactor(dta, k_factors=2, factor_order=1)
    mod.update(mod.start_params)
    sim_cfa = mod.simulation_smoother(method='cfa')
    res = mod.ssm.smooth()
    sim_cfa.simulate(np.zeros((mod.k_states, mod.nobs)))
    assert_allclose(sim_cfa.simulated_state, res.smoothed_state)