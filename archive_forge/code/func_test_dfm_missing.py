import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
def test_dfm_missing(reset_randomstate):
    endog = np.random.normal(size=(100, 3))
    endog[0, :1] = np.nan
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod.ssm.filter_collapsed = True
    res = mod.smooth(mod.start_params)
    mod.ssm.filter_collapsed = False
    res2 = mod.smooth(mod.start_params)
    assert_allclose(res.llf, res2.llf)