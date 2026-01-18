from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
def model_dfm(endog=None, params=None, factor_order=2):
    if endog is None:
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
    if params is None:
        params = np.r_[0.5, 1.0, 1.5, 2.0, 0.9, 0.1]
    mod = DynamicFactor(endog, k_factors=1, factor_order=factor_order)
    mod.update(params)
    ssm = mod.ssm
    ssm.filter_univariate = True
    init = Initialization(ssm.k_states, 'diffuse')
    ssm.initialize(init)
    return (mod, ssm)