import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_memory_no_predicted():
    endog = [0.5, 1.2, 0.4, 0.6]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res1 = mod.filter([0.5, 1.0])
    mod.ssm.memory_no_predicted = True
    res2 = mod.filter([0.5, 1.0])
    assert_equal(res1.predicted_state.shape, (1, 5))
    assert_(res2.predicted_state is None)
    assert_equal(res1.predicted_state_cov.shape, (1, 1, 5))
    assert_(res2.predicted_state_cov is None)
    assert_raises(ValueError, res2.predict, dynamic=True)
    assert_raises(ValueError, res2.get_prediction, dynamic=True)
    assert_allclose(res1.forecast(10), res2.forecast(10))
    fcast1 = res1.get_forecast(10)
    fcast2 = res1.get_forecast(10)
    assert_allclose(fcast1.summary_frame(), fcast2.summary_frame())