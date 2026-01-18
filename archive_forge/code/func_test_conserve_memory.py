import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_conserve_memory(self):
    model = self.model
    model.conserve_memory = MEMORY_STORE_ALL
    model.memory_no_forecast = True
    assert_equal(model.conserve_memory, MEMORY_NO_FORECAST)
    model.memory_no_filtered = True
    assert_equal(model.conserve_memory, MEMORY_NO_FORECAST | MEMORY_NO_FILTERED)
    model.memory_no_forecast = False
    assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)
    model.set_conserve_memory(MEMORY_NO_PREDICTED)
    assert_equal(model.conserve_memory, MEMORY_NO_PREDICTED)
    model.set_conserve_memory(memory_no_filtered=True, memory_no_predicted=False)
    assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)
    model.conserve_memory = 0
    for name in model.memory_options:
        if name == 'memory_conserve':
            continue
        setattr(model, name, True)
    assert_equal(model.conserve_memory, MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED | MEMORY_NO_LIKELIHOOD | MEMORY_NO_GAIN | MEMORY_NO_SMOOTHING | MEMORY_NO_STD_FORECAST)
    assert_equal(model.conserve_memory & MEMORY_CONSERVE, MEMORY_CONSERVE)
    for name in model.memory_options:
        if name == 'memory_conserve':
            continue
        setattr(model, name, False)
    assert_equal(model.conserve_memory, 0)