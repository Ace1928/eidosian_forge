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
def model_common_level(endog=None, params=None, restricted=False):
    if endog is None:
        y11 = 10.2394
        y21 = 8.2304
        endog = np.column_stack([np.r_[y11, [1] * 9], np.r_[y21, [1] * 9]])
    if params is None:
        params = [0.1111, 3.2324]
    theta, sigma2_mu = params
    if not restricted:
        ssm = KalmanSmoother(k_endog=2, k_states=2, k_posdef=1)
        ssm.bind(endog.T)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design'] = np.array([[1, 0], [theta, 1]])
        ssm['obs_cov'] = np.eye(2)
        ssm['transition'] = np.eye(2)
        ssm['selection', 0, 0] = 1
        ssm['state_cov', 0, 0] = sigma2_mu
    else:
        ssm = KalmanSmoother(k_endog=2, k_states=1, k_posdef=1)
        ssm.bind(endog.T)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design'] = np.array([[1, theta]]).T
        ssm['obs_cov'] = np.eye(2)
        ssm['transition', :] = 1
        ssm['selection', :] = 1
        ssm['state_cov', :] = sigma2_mu
    return ssm