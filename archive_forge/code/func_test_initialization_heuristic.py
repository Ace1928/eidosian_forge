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
def test_initialization_heuristic(oildata):
    model_estimated = ETSModel(oildata, error='add', trend='add', damped_trend=True, initialization_method='estimated')
    model_heuristic = ETSModel(oildata, error='add', trend='add', damped_trend=True, initialization_method='heuristic')
    fit_estimated = model_estimated.fit(disp=False)
    fit_heuristic = model_heuristic.fit(disp=False)
    yhat_estimated = fit_estimated.fittedvalues.values
    yhat_heuristic = fit_heuristic.fittedvalues.values
    assert_allclose(yhat_estimated[10:], yhat_heuristic[10:], rtol=0.5)