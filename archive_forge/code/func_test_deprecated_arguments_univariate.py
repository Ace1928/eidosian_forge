import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
def test_deprecated_arguments_univariate():
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True, initialization='diffuse')
    mod.update([intercept, 0.5, 2.0])
    mds = np.arange(10) / 10.0
    sds = np.arange(10)[::-1] / 20.0
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(1))
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(disturbance_variates=np.r_[mds, sds], initial_state_variates=np.zeros(1))
    actual = sim.simulated_state[0]
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(1), pretransformed_measurement_disturbance_variates=True, pretransformed_state_disturbance_variates=True)
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, pretransformed=True, initial_state_variates=np.zeros(1))
    actual = sim.simulated_state[0]
    assert_allclose(actual, desired)