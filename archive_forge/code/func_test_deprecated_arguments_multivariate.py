import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_deprecated_arguments_multivariate():
    endog = np.array([[0.3, 1.4], [-0.1, 0.6], [0.2, 0.7], [0.1, 0.9], [0.5, -0.1]])
    mod = varmax.VARMAX(endog, order=(1, 0, 0))
    mod.update([1.2, 0.5, 0.8, 0.1, -0.2, 0.5, 5.2, 0.5, 8.1])
    mds = np.arange(10).reshape(5, 2) / 10.0
    sds = np.arange(10).reshape(5, 2)[::-1] / 20.0
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(2))
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(disturbance_variates=np.r_[mds.ravel(), sds.ravel()], initial_state_variates=np.zeros(2))
    actual = sim.simulated_state[0]
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(2), pretransformed_measurement_disturbance_variates=True, pretransformed_state_disturbance_variates=True)
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, pretransformed=True, initial_state_variates=np.zeros(2))
    actual = sim.simulated_state[0]
    assert_allclose(actual, desired)