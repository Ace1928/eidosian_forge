from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.mark.parametrize('repetitions', [1, 10])
@pytest.mark.parametrize('random_errors', [None, 'bootstrap'])
def test_forecast_1_simulation(austourists, random_errors, repetitions):
    fit = ExponentialSmoothing(austourists, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, initialization_method='estimated').fit()
    sim = fit.simulate(1, anchor=0, random_errors=random_errors, repetitions=repetitions)
    expected_shape = (1,) if repetitions == 1 else (1, repetitions)
    assert sim.shape == expected_shape
    sim = fit.simulate(10, anchor=0, random_errors=random_errors, repetitions=repetitions)
    expected_shape = (10,) if repetitions == 1 else (10, repetitions)
    assert sim.shape == expected_shape