import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def check_equivalent_models(mod, mod2):
    attrs = ['k_factors', 'factor_order', 'error_order', 'error_var', 'error_cov_type', 'enforce_stationarity', 'mle_regression', 'k_params']
    ssm_attrs = ['nobs', 'k_endog', 'k_states', 'k_posdef', 'obs_intercept', 'design', 'obs_cov', 'state_intercept', 'transition', 'selection', 'state_cov']
    for attr in attrs:
        assert_equal(getattr(mod2, attr), getattr(mod, attr))
    for attr in ssm_attrs:
        assert_equal(getattr(mod2.ssm, attr), getattr(mod.ssm, attr))
    assert_equal(mod2._get_init_kwds(), mod._get_init_kwds())