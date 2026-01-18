import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_manual_approximate_diffuse_initialization():
    endog = results_sarimax.wpi1_data
    mod1 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    mod1.ssm.initialize_approximate_diffuse(1000000000.0)
    res1 = mod1.filter([0.5, 0.2, 0.1, 1])
    mod2 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    mod2.ssm.initialize_known(res1.filter_results.initial_state, res1.filter_results.initial_state_cov)
    res2 = mod2.filter([0.5, 0.2, 0.1, 1])
    mod3 = sarimax.SARIMAX(endog, order=(3, 0, 0), initialization='known', initial_state=res1.filter_results.initial_state, initial_state_cov=res1.filter_results.initial_state_cov)
    res3 = mod3.filter([0.5, 0.2, 0.1, 1])
    mod4 = sarimax.SARIMAX(endog, order=(3, 0, 0), initialization='approximate_diffuse', initial_variance=1000000000.0)
    res4 = mod4.filter([0.5, 0.2, 0.1, 1])
    assert_almost_equal(res1.llf, res2.llf)
    assert_almost_equal(res1.filter_results.filtered_state, res2.filter_results.filtered_state)
    assert_almost_equal(res1.llf, res3.llf)
    assert_almost_equal(res1.filter_results.filtered_state, res3.filter_results.filtered_state)
    assert_almost_equal(res1.llf, res4.llf)
    assert_almost_equal(res1.filter_results.filtered_state, res4.filter_results.filtered_state)