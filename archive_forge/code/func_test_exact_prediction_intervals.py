from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_exact_prediction_intervals(austourists_model_fit):
    fit = austourists_model_fit._results

    class DummyModel:

        def __init__(self, short_name):
            self.short_name = short_name
    fit.damping_trend = 1 - 0.001
    fit.model = DummyModel('AAdN')
    steps = 5
    s_AAdN = fit._relative_forecast_variance(steps)
    fit.model = DummyModel('AAN')
    s_AAN = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdN, s_AAN, 2)
    fit.damping_trend = 1 - 0.001
    fit.model = DummyModel('AAdA')
    steps = 5
    s_AAdA = fit._relative_forecast_variance(steps)
    fit.model = DummyModel('AAA')
    s_AAA = fit._relative_forecast_variance(steps)
    assert_almost_equal(s_AAdA, s_AAA, 2)