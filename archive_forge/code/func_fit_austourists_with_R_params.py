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
def fit_austourists_with_R_params(model, results_R, set_state=False):
    """
    Fit the model with params as found by R's forecast package
    """
    params = get_params_from_R(results_R)
    with model.fix_params(dict(zip(model.param_names, params))):
        fit = model.fit(disp=False)
    if set_state:
        states_R = get_states_from_R(results_R, model._k_states)
        fit.states = states_R
    return fit