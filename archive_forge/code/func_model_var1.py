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
def model_var1(endog=None, params=None, measurement_error=False, init=None):
    if endog is None:
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
    if params is None:
        params = np.r_[0.5, 0.3, 0.2, 0.4, 2 ** 0.5, 0, 3 ** 0.5]
        if measurement_error:
            params = np.r_[params, 4, 5]
    mod = VARMAX(endog, order=(1, 0), trend='n', measurement_error=measurement_error)
    mod.update(params)
    ssm = mod.ssm
    if init is None:
        init = Initialization(ssm.k_states, 'diffuse')
    ssm.initialize(init)
    return (mod, ssm)